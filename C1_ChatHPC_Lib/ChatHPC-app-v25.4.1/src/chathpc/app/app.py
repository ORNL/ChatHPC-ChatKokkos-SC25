"""App Module: used to construct an app for training ChatHPC LLMs."""

from __future__ import annotations

import atexit
import os
import readline
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import torch
from loguru import logger
from peft import (
    LoraConfig,  # type: ignore
    PeftModel,  # type: ignore
    get_peft_model,  # type: ignore
    prepare_model_for_kbit_training,  # type: ignore
)
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, JsonConfigSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict
from pytz import timezone
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

import chathpc
import chathpc.app
from chathpc.app.ollama_interface import ollama_chat_evaluate
from chathpc.app.openai_interface import ChatHPCOpenAI
from chathpc.app.utils import template_utils
from chathpc.app.utils.common_utils import load_json_yaml_arg, run
from chathpc.app.utils.datastore import save_json
from chathpc.app.utils.template_utils import map_keywords
from chathpc.app.utils.verify_utils import ignore_minor

DEFAULT_APP_CONFIG_FILE = Path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "config/default_app_settings.json"))
)


class AppConfig(BaseSettings):
    """Configuration settings for the ChatHPC application.

    This class inherits from Pydantic BaseSettings to manage application configuration
    through multiple sources with a defined priority order.

    Attributes:
        data_file (Path): Training data JSON file path.
        base_model_path (Path): Pre-trained base LLM model directory.
        finetuned_model_path (Path): Directory for fine-tuned model layers.
        merged_model_path (Path): Directory for complete merged model.
        training_output_dir (Path): Directory for training output and checkpoints.
        max_training_tokens (int): Maximum tokens for training set tokenization.
        max_response_tokens (int): Maximum tokens for model response generation.
        prompt_history_file (Path): File path for interactive chat history.
        prompt_template_file (Path): File containing prompt template for training/inference.
        prompt_template (str): Direct string template for prompts.
        use_wandb (bool): Enable/disable Weights & Biases logging.

    Configuration Priority:
        1. Environment variables (CHATHPC_ prefix)
        2. .env file
        3. Direct initialization
        4. JSON config file
        5. File secrets

    Example:
        ```python
        config = AppConfig(base_model_path="/path/to/model")
        config = AppConfig.from_json("config.json")
        ```

    Note:
        - All paths are handled as Path objects
        - UTF-8 encoding used for all file operations
        - Either prompt_template_file or prompt_template must be set
    """

    data_file: Path = Field(..., description="Path to the JSON file containing training data for model fine-tuning.")
    base_model_path: Path = Field(
        Path("/auto/projects/ChatHPC/models/cache/meta-llama/CodeLlama-7b-hf"),
        description="Path to the pre-trained base LLM model directory.",
    )
    finetuned_model_path: Path = Field(
        Path("peft_adapter"), description="Path where fine-tuned model layers will be saved."
    )
    merged_model_path: Path = Field(
        Path("merged_adapters"), description="Path where the complete merged model will be saved."
    )
    training_output_dir: Path = Field(
        Path("training_checkpoints"), description="Path where training output will be saved."
    )
    max_training_tokens: int = Field(
        512, gt=0, description="Maximum number of tokens to use to tokenize the training sets."
    )
    max_response_tokens: int = Field(600, gt=0, description="Maximum number of tokens to generate in model responses.")
    prompt_history_file: Path = Field(
        Path("~/.chathpc_history"), description="Path to the file containing interactive prompt history."
    )
    prompt_template_file: Path | None = Field(
        None, description="Path to the prompt template to use for training and inference."
    )
    prompt_template: str | None = Field(
        None, description="Path to the prompt template to use for training and inference."
    )
    use_wandb: bool = Field(False, description="Whether to use Weights & Biases for logging.")

    model_config = SettingsConfigDict(
        # cli_parse_args=True,
        env_prefix="CHATHPC_",
        env_file=".env",
        env_file_encoding="utf-8",
        # json_file=DEFAULT_APP_CONFIG_FILE,
        json_file_encoding="utf-8",
        extra="allow",
    )

    @model_validator(mode="before")
    @classmethod
    def check_for_prompt_template(cls, values):
        """Validate prompt template configuration.

        This validator ensures that exactly one of prompt_template_file or prompt_template
        is set in the configuration. Having both or neither is invalid.

        Args:
            values (dict): Dictionary of configuration values to validate.

        Returns:
            dict: The validated configuration values.

        Raises:
            ValueError: If neither or both prompt template options are set.

        Example:
            Valid configurations:
            - prompt_template_file set, prompt_template None
            - prompt_template set, prompt_template_file None

            Invalid configurations:
            - Both prompt_template and prompt_template_file set
            - Neither prompt_template nor prompt_template_file set
        """
        if not (bool(values.get("prompt_template_file")) | bool(values.get("prompt_template"))):
            raise ValueError("Either prompt_template_file or prompt_template must be set.")
        if bool(values.get("prompt_template_file")) & bool(values.get("prompt_template")):
            raise ValueError("prompt_template_file and prompt_template should not both be set.")
        return values

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            dotenv_settings,
            init_settings,
            JsonConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    @classmethod
    def from_json(cls, json_or_file: str | Path | dict, extra_params: str | Path | dict | None = None) -> AppConfig:
        """Create an AppConfig instance from JSON configuration sources.

        This class method creates an AppConfig instance by combining settings from a primary
        JSON source and optional additional parameters.

        Args:
            json_or_file (Union[str, Path, dict]): Primary configuration source - either a
                path to a JSON file or a dictionary with configuration values.
            extra_params (Union[str, Path, dict], optional): Additional configuration source
                to override or supplement primary settings.

        Returns:
            AppConfig: A new AppConfig instance initialized with combined settings.

        Example:
            ```python
            # From JSON file
            config = AppConfig.from_json("config.json")

            # With extra parameters
            config = AppConfig.from_json("config.json", {"max_response_tokens": 800})

            # From dictionary
            config = AppConfig.from_json({"data_file": "data.json"})
            ```

        Note:
            When both sources are provided, settings from extra_params override
            corresponding values from the primary source.
        """
        json_config = load_json_yaml_arg(json_or_file)
        extra_config = load_json_yaml_arg(extra_params)
        json_config.update(extra_config)
        return cls(**json_config)


class App:
    """Main application class for ChatHPC Application.

    This class handles the initialization, loading, and management of models,
    datasets, and training processes for the ChatHPC application. It provides
    methods for loading different types of models, evaluating prompts, and
    fine-tuning the model.

    Attributes:
        config (AppConfig): Configuration settings for the application.
        tokenizer: Tokenizer for processing input text.
        model: The language model used for text generation and fine-tuning.
        train_dataset: Dataset used for training.
        eval_dataset: Dataset used for evaluation.
        tokenized_train_dataset: Tokenized version of training dataset.
        tokenized_val_dataset: Tokenized version of validation dataset.
        peft_config: Configuration for LoRA fine-tuning.
        training_args: Arguments for model training.

    Methods:
        load_base_model(): Loads the base LLM model.
        load_finetuned_model(): Loads a model with fine-tuned layers.
        load_merged_model(): Loads a complete merged model.
        load_datasets(): Loads training and evaluation datasets.
        evaluate_model(): Generates responses for given prompts.
        chat_prompt(): Creates formatted prompts for questions.
        chat_evaluate(): Evaluates questions with context.
        tokenize_training_set(): Prepares datasets for training.
        train(): Executes model fine-tuning process.
        interactive(): Starts interactive chat session.
        print_config(): Displays current configuration settings.
    """

    def __init__(self, app_config: AppConfig | None = None):
        """Initialize the ChatHPC application instance.

        This method sets up a new application instance with configuration settings
        and initializes the Jinja2 environment for template processing.

        Args:
            app_config (AppConfig, optional): Application configuration settings.
                If None, creates default AppConfig instance.

        Sets:
            - self.config: Application configuration settings
            - self.jinja: Jinja2 environment for template processing

        Example:
            ```python
            # With default settings
            app = App()

            # With custom settings
            config = AppConfig(base_model_path="/path/to/model")
            app = App(app_config=config)
            ```

        Note:
            Model loading and other initializations must be performed explicitly
            by calling the appropriate methods after initialization.
        """
        if app_config is None:
            app_config = AppConfig()  # type: ignore

        self.config = app_config

        self.jinja = jinja2.Environment(autoescape=False, keep_trailing_newline=True)  # noqa: S701
        self._load_templates()

    @classmethod
    def from_json(cls, json_or_file: str | Path | dict, extra_params: str | Path | dict | None = None) -> App:
        """Create an App instance from JSON configuration sources.

        This class method creates an App instance by combining settings from a primary
        JSON source and optional additional parameters.

        Args:
            json_or_file (Union[str, Path, dict]): Primary configuration source - either a
                path to a JSON file or a dictionary with configuration values.
            extra_params (Union[str, Path, dict], optional): Additional configuration source
                to override or supplement primary settings.

        Returns:
            App: A new App instance initialized with combined settings.

        Example:
            ```python
            # From JSON file
            app = App.from_json("config.json")

            # With extra parameters
            app = App.from_json("config.json", {"max_response_tokens": 800})

            # From dictionary
            app = App.from_json({"data_file": "data.json"})
            ```

        Note:
            When both sources are provided, settings from extra_params override
            corresponding values from the primary source.
        """
        config = AppConfig.from_json(json_or_file, extra_params=extra_params)
        return cls(app_config=config)

    def _load_templates(self):
        """Load and initialize prompt templates for training and inference.

        This method loads prompt templates either from a file or a string configuration,
        processes them for training and inference use, and initializes Jinja2 templates.

        The templates are split into prefix and postfix components around the response
        section for proper formatting during training and inference.

        Raises:
            ValueError: If neither prompt_template nor prompt_template_file is properly configured
            ValueError: If the specified prompt template file does not exist

        Sets:
            - self.training_template: Complete Jinja2 template for training
            - self.inference_template: Prefix template for inference
            - self.postfix_template: Postfix template for inference
            - self._prompt_prefix: Raw prefix string
            - self._prompt_postfix: Raw postfix string

        Example:
            ```python
            app = App(config)
            app._load_templates()  # Templates are loaded during initialization
            ```

        Note:
            This method is called automatically during App initialization and should
            not typically be called directly.
        """
        relative_path = None
        if (
            hasattr(self.config, "filename")
            and self.config.prompt_template is None
            and self.config.prompt_template_file is not None
            and not self.config.prompt_template_file.is_absolute()
        ):
            filename = Path(self.config.filename)  # type: ignore
            relative_path = filename.parent / self.config.prompt_template_file

        if self.config.prompt_template is not None:
            prompt_template_string = self.config.prompt_template

        else:
            if self.config.prompt_template_file is None:
                raise ValueError("Unexpected Error: Prompt template file is not set.")

            if self.config.prompt_template_file.is_file():
                logger.info("Loading prompt template from {file}", file=self.config.prompt_template_file)
                with open(self.config.prompt_template_file) as f:
                    prompt_template_string = f.read()
            elif relative_path is not None and relative_path.is_file():
                logger.info("Loading prompt template from {file}", file=relative_path)
                with open(relative_path) as f:
                    prompt_template_string = f.read()
            else:
                raise ValueError("Prompt template file not found.")

        prompt_template_string = template_utils.normalize_template(prompt_template_string)
        self.config.prompt_template = prompt_template_string

        self.training_template = self.jinja.from_string(prompt_template_string)
        self._prompt_prefix, self._prompt_postfix = template_utils.split_on_response(prompt_template_string)
        self.inference_template = self.jinja.from_string(self._prompt_prefix)
        self.postfix_template = self.jinja.from_string(self._prompt_postfix)

    def load_base_model(self) -> None:
        """Load and initialize the base Large Language Model.

        This method initializes both the tokenizer and model from the base model path
        specified in the application preferences. The model is loaded with specific
        configurations for optimal performance.

        Requires:
            - preferences.base_model_path must be set to a valid model path

        Sets:
            - self.tokenizer: Initialized AutoTokenizer for text processing
            - self.model: Initialized AutoModelForCausalLM in float16 precision

        Example:
            ```python
            >>> app = App()
            >>> app.preferences.base_model_path = "path/to/model"
            >>> app.load_base_model()
            ```

        Note:
            The model is loaded with float16 precision and automatic device mapping
            for optimal performance on available hardware.
        """

        logger.info("Loading the base model from {path}", path=self.config.base_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            # device_map={'':torch.cuda.current_device()}
        )

    def load_finetuned_model(self) -> None:
        """Load and initialize the finetuned Large Language Model.

        This method loads a finetuned model by first initializing the base model and tokenizer,
        then loading the finetuned layers on top of it using PeftModel.

        Requires:
            - preferences.base_model_path must be set to a valid base model path
            - preferences.finetuned_model_path must be set to a valid finetuned model path

        Sets:
            - self.tokenizer: Initialized AutoTokenizer for text processing
            - self.model: Initialized PeftModel with finetuned layers

        Example:
            ```python
            >>> app = App()
            >>> app.preferences.base_model_path = "path/to/base/model"
            >>> app.preferences.finetuned_model_path = "path/to/finetuned/model"
            >>> app.load_finetuned_model()
            ```

        Note:
            This method first calls load_base_model() to initialize the foundation model
            before applying the finetuned layers.
        """

        logger.info("Loading the finetuned model from {path}", path=self.config.finetuned_model_path)

        self.load_base_model()

        self.model = PeftModel.from_pretrained(self.model, self.config.finetuned_model_path)  # type: ignore

    def load_merged_model(self) -> None:
        """Load and initialize the merged Large Language Model.

        This method loads a complete merged model that combines the base model with
        finetuned layers into a single model file. The tokenizer is initialized from
        the base model path while the full model is loaded from the merged model path.

        Requires:
            - preferences.base_model_path must be set to a valid base model path for tokenizer
            - preferences.merged_model_path must be set to a valid merged model path

        Sets:
            - self.tokenizer: Initialized AutoTokenizer for text processing
            - self.model: Initialized AutoModelForCausalLM with merged weights

        Example:
            ```python
            >>> app = App()
            >>> app.preferences.base_model_path = "path/to/base/model"
            >>> app.preferences.merged_model_path = "path/to/merged/model"
            >>> app.load_merged_model()
            ```

        Note:
            The model is loaded with float16 precision and automatic device mapping
            for optimal performance on available hardware.
        """

        logger.info("Loading the merged model from {path}", path=self.config.merged_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.merged_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            # device_map={'':torch.cuda.current_device()}
        )

    def load_datasets(self) -> None:
        """Load training and evaluation datasets from a JSON file.

        This method loads datasets from the JSON file specified in the application preferences.
        The datasets are loaded using the Hugging Face datasets library and split into
        training and evaluation sets.

        Config:
            preferences.data_file (str): Path to the JSON file containing the datasets.

        Sets:
            self.train_dataset: Dataset object for training
            self.eval_dataset: Dataset object for evaluation

        Requires:
            - The data file must be in JSON format
            - The data file path must be set in preferences.data_file
        """
        logger.info("Loading the dataset from {path}", path=self.config.data_file)

        from datasets import load_dataset

        self.train_dataset = load_dataset("json", data_files=self.config.data_file.as_posix(), split="train")
        self.eval_dataset = load_dataset("json", data_files=self.config.data_file.as_posix(), split="train")

    def evaluate_model(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Generate a model response for a given input prompt.

        Args:
            prompt (str): Input text prompt for model evaluation.
            max_new_tokens (int|None): Maximum tokens to generate. Defaults to config.max_response_tokens.

        Returns:
            str: Generated text response from the model.

        Requires:
            - Initialized model via one of:
                - load_base_model()
                - load_finetuned_model()
                - load_merged_model()
            - Initialized tokenizer

        Example:
            ```python
            app = App()
            app.load_base_model()
            response = app.evaluate_model("What is Kokkos?", max_new_tokens=100)
            print(response)  # "Kokkos is a programming model..."
            ```

        Note:
            Uses evaluation mode and torch.no_grad() for inference.
            Input is processed on CUDA if available.
        """
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        if max_new_tokens is None:
            max_new_tokens = self.config.max_response_tokens

        self.model.eval()  # type: ignore
        with torch.no_grad():
            output = self.model.generate(  # type: ignore
                **model_input, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id
            )[0]
            return self.tokenizer.decode(output)

    def chat_prompt(self, **kwargs) -> str:
        """Create a formatted prompt for chat questions.

        This method generates a structured prompt using the inference template by combining
        provided keyword arguments according to the template defined in the application
        configuration.

        Args:
            **kwargs: Keyword arguments to be passed to the template.
                Common arguments include:
                - question (str): The question to be answered
                - context (str): Supporting context or documentation
                Additional arguments can be used if defined in the template.

        Returns:
            str: A formatted prompt string following the inference template.

        Requires:
            - Initialized inference_template via _load_templates()
            - Template must be properly formatted with expected variables

        Example:
            ```python
            app = App()
            prompt = app.chat_prompt(
                question="How do I use Views?",
                context="Views are memory spaces in Kokkos...",
            )
            print(prompt)  # Returns formatted prompt based on template
            ```

        Note:
            - The actual prompt format is determined by the inference template loaded during initialization
            - Keywords are automatically mapped using template_utils.map_keywords()
            - This method is typically used internally by chat_evaluate()
        """

        return self.inference_template.render(**template_utils.map_keywords(kwargs))

    def chat_evaluate(self, **kwargs) -> str:
        """Evaluate a question with provided context using the model.

        This method processes a question-context pair through the model by:
        1. Formatting the input using the inference template
        2. Generating a response using the model
        3. Returning both the response and original prompt

        Args:
            question (str): The question to be answered by the model.
            **kwargs: Additional keyword arguments passed to evaluate_model().
                Common arguments include:
                - max_new_tokens (int): Override default token generation limit
                - Other template variables defined in prompt template

        Returns:
            str: Generated model response.

        Requires:
            - Initialized model via one of load methods:
                - load_base_model()
                - load_finetuned_model()
                - load_merged_model()
            - Initialized tokenizer and templates

        Example:
            ```python
            app = App()
            app.load_merged_model()
            response = app.chat_evaluate(
                question="What is Kokkos?",
                context="Kokkos is a performance portable programming model...",
                max_new_tokens=200,
            )
            print(response)  # Prints model's explanation of Kokkos
            ```

        Note:
            - Uses chat_prompt() for template-based input formatting
            - Uses evaluate_model() for response generation
            - Response format follows inference template structure
            - Template variables can be passed via kwargs
        """
        prompt = self.chat_prompt(**kwargs)
        return self.evaluate_model(prompt)

    def chat_evaluate_extract(self, **kwargs) -> str:
        """Extract the model's answer from a chat evaluation response.

        This method combines chat_evaluate() with answer extraction, removing template
        formatting and returning only the model's direct response.

        Args:
            **kwargs: Keyword arguments passed to chat_evaluate().
                Common arguments include:
                - question (str): The question to be answered
                - context (str): Supporting context or documentation
                - max_new_tokens (int): Override default token generation limit
                Additional arguments can be used if defined in the template.

        Returns:
            str: The extracted answer from the model's response, without template formatting.

        Example:
            ```python
            app = App()
            app.load_merged_model()
            answer = app.chat_evaluate_extract(
                question="What is Kokkos?", context="Kokkos is a programming model..."
            )
            print(answer)  # Prints just the model's answer without template
            ```

        Note:
            - Uses chat_evaluate() to generate the full response
            - Automatically extracts the answer portion using template structure
            - More concise than chat_evaluate() for direct answer retrieval
        """
        chat_response = self.chat_evaluate(**kwargs)
        return self.extract_answer(chat_response, **kwargs)

    def training_prompt(self, **kwargs) -> str:
        """Create a formatted prompt for training data.

        This method generates a structured prompt using the training template by combining
        provided keyword arguments according to the template defined in the application
        configuration.

        Args:
            **kwargs: Keyword arguments to be passed to the template.
                Common arguments include:
                - question (str): The question to be used in training
                - context (str): Supporting context or documentation
                - answer (str): The expected answer or response
                Additional arguments can be used if defined in the template.

        Returns:
            str: A formatted prompt string following the training template.

        Requires:
            - Initialized training_template via _load_templates()
            - Template must be properly formatted with expected variables

        Example:
            ```python
            app = App()
            prompt = app.training_prompt(
                question="How do I use Views?",
                context="Views are memory spaces in Kokkos...",
                answer="To use Views in Kokkos...",
            )
            print(prompt)  # Returns formatted prompt based on template
            ```

        Note:
            - The actual prompt format is determined by the training template loaded during initialization
            - Keywords are automatically mapped using template_utils.map_keywords()
            - This method is typically used internally by tokenize_training_set()
        """

        return self.training_template.render(**template_utils.map_keywords(kwargs))

    def tokenize_training_set(self) -> None:
        """Tokenize the training and validation datasets.

        This method processes the loaded datasets by tokenizing text data for model training.
        It handles padding configuration and EOS token management during tokenization.

        Requires:
            - Initialized datasets via load_datasets()
            - Initialized tokenizer via loading a model

        Sets:
            - self.tokenized_train_dataset: Processed training dataset
            - self.tokenized_val_dataset: Processed validation dataset

        Example:
            ```python
            app = App()
            app.load_base_model()
            app.load_datasets()
            app.tokenize_training_set()
            ```

        Note:
            - The method uses the training_prompt template from config to format inputs before tokenization.
            - This method also handles padding token configuration and adds/removes EOS tokens as needed for the tokenization process.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        def tokenize(prompt):
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.config.max_training_tokens,
                padding=False,
                return_tensors=None,
            )
            result_full = self.tokenizer(
                prompt,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            if result != result_full:
                logger.warning(
                    "Training tokenizer needs {token_count} tokens to fully tokenize the training input and max training tokens is set to {max_training_tokens}. \nPrompt: {prompt}\nCropped to: {prompt_cropped}",
                    token_count=len(result_full.data["input_ids"]),
                    max_training_tokens=self.config.max_training_tokens,
                    prompt=prompt,
                    prompt_cropped=self.tokenizer.decode(result.data["input_ids"]),
                )

            # "self-supervised learning" means the labels are also the inputs:
            result["labels"] = result["input_ids"].copy()  # type: ignore

            return result

        def generate_and_tokenize_prompt(data_point):
            full_prompt = self.training_prompt(**data_point)
            return tokenize(full_prompt)

        self.tokenizer.add_eos_token = True

        self.tokenized_train_dataset = self.train_dataset.map(generate_and_tokenize_prompt)
        self.tokenized_val_dataset = self.eval_dataset.map(generate_and_tokenize_prompt)

        self.tokenizer.add_eos_token = False

    def train(self):
        """Train the model using fine-tuning layers.

        This method performs fine-tuning of the base model using LoRA (Low-Rank Adaptation)
        configuration. It prepares the model for training, sets up training arguments,
        and executes the training process.

        Requires:
            - App.load_datasets() must be called first to load training data
            - App.load_base_model() must be called first to load the base model
            - Tokenizer and model must be properly initialized

        Sets:
            - self.peft_config: LoRA configuration for fine-tuning
            - self.training_args: Training arguments for the Trainer
            - self.model: Updated model after training

        Saves:
            - Finetuned model layers to preferences.finetuned_model_path
            - Complete merged model to preferences.merged_model_path

        Note:
            This method uses Hugging Face's Trainer for the training process and
            supports multi-GPU training when available. It also integrates with
            Weights & Biases (wandb) for experiment tracking.
        """

        self.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        )
        self.model.train()  # type: ignore # put model back into training mode
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

        batch_size = 128
        per_device_train_batch_size = 32
        gradient_accumulation_steps = batch_size // per_device_train_batch_size
        output_dir = self.config.training_output_dir.as_posix()

        # resume_from_checkpoint = os.path.join(base_model_path, "pytorch_model-00001-of-00003.bin")

        # if resume_from_checkpoint:
        #     if os.path.exists(resume_from_checkpoint):
        #         print(f"Restarting from {resume_from_checkpoint}")
        #         adapters_weights = torch.load(resume_from_checkpoint)
        #         set_peft_model_state_dict(self.model, adapters_weights)
        #     else:
        #         print(f"Checkpoint {resume_from_checkpoint} not found")

        wandb_project = "ChatHPC"
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project

        if torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            print("multiple gpus detected!")
            self.model.is_parallelizable = True  # type: ignore
            self.model.model_parallel = True  # type: ignore

        self.training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            max_steps=400,
            # max_steps=20,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps",  # if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=20,
            save_steps=20,
            output_dir=output_dir,
            # save_total_limit=3,
            load_best_model_at_end=False,
            # ddp_find_unused_parameters=False if ddp else None,
            group_by_length=True,  # group sequences of roughly the same length together to speed up training
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=f"codellama-{datetime.now(tz=timezone('EST')).strftime('%Y-%m-%d-%H-%M')}",  # if use_wandb else None,
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,  # type: ignore
            eval_dataset=self.tokenized_val_dataset,  # type: ignore
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )

        self.model.config.use_cache = False  # type: ignore

        # old_state_dict = model.state_dit
        # model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        #     model, type(model)
        # )

        if torch.__version__ >= "2" and sys.platform != "win32":
            print("compiling the model")
            self.model = torch.compile(self.model)

        # model.to('cuda')

        trainer.train()

        print("Saving Model...")
        trainer.model.save_pretrained(self.config.finetuned_model_path)  # type: ignore
        self.save_readme(self.config.finetuned_model_path)
        print("Merging model...")
        self.model = trainer.model.merge_and_unload()  # type: ignore
        print("Saving merged model...")
        self.tokenizer.save_pretrained(self.config.merged_model_path)
        self.model.save_pretrained(self.config.merged_model_path)
        print("Saving README.md...")
        self.save_readme(self.config.merged_model_path)

    def interactive(self, args, prompt="chathpc") -> None:
        """Start an interactive chat session with the model.

        This method provides a command-line interface for interacting with the model.
        It maintains a command history and supports context setting for conversations.

        Commands:
            /bye: Exit the interactive session
            /context: Set a new context for subsequent questions

        Args:
            prompt (str, optional): The prompt prefix to display. Defaults to "chathpc".

        Requires:
            - A model must be loaded via one of:
                - load_base_model()
                - load_finetuned_model()
                - load_merged_model()
            - The tokenizer must be initialized

        Example:
            ```python
            >>> app = App()
            >>> app.load_merged_model()
            >>> app.interactive()
            chathpc ()> What is Kokkos?
        """
        history_file = self.config.prompt_history_file.expanduser().as_posix()
        try:
            readline.read_history_file(history_file)
            h_len = readline.get_current_history_length()
        except FileNotFoundError:
            open(history_file, "wb+").close()
            readline.add_history("/context")
            readline.add_history("/bye")
            h_len = readline.get_current_history_length()

        def save_history(prev_h_len, histfile):
            new_h_len = readline.get_current_history_length()
            readline.set_history_length(1000)
            readline.append_history_file(new_h_len - prev_h_len, histfile)

        atexit.register(save_history, h_len, history_file)

        context = None
        print("Use '/bye' to exit.\nUse '/context' to set context.")
        while True:
            prompt_line = f"{prompt} ({context})> " if context is not None else f"{prompt}> "
            user_input = input(prompt_line)
            if user_input == "/bye":
                print("Goodbye!")
                break
            if user_input.startswith("/context"):
                context = user_input.replace("/context", "").strip()
                if context == "":
                    context = input("Context: ")
                if context.strip() == "":
                    context = None
                continue
            if args.extract:
                print(self.chat_evaluate_extract(question=user_input, context=context))
            else:
                print(self.chat_evaluate(question=user_input, context=context))

    def verify(
        self,
        save_verify_data_path: str | Path | None = None,
        ollama_model: str | None = None,
        openai_model: str | None = None,
    ) -> int:
        """Verify model outputs against the training dataset.

        This method runs verification tests on the model by comparing its outputs
        against the training set.

        Args:
            save_verify_data_path (Union[str, Path, None]): Optional path to save
                verification results.
            ollama_model (str, optional): Name of Ollama model, if using Ollama instead of app's model

        Returns:
            int: The number of errors.

        Example:
            ```python
            app = App()
            app.load_merged_model()
            tests_failed = app.verify(save_verify_data_path="verify_results.json")
            ```
        """
        verify_data = []

        if ollama_model is not None and openai_model is not None:
            raise RuntimeError("Both Ollama model and OpenAI model cannot both be set. Only one should be set.")

        openai_client = ChatHPCOpenAI(self.config) if openai_model is not None else None

        for i, item in tqdm(enumerate(self.train_dataset), "Verify", total=len(self.train_dataset)):  # type: ignore
            item_mapped = map_keywords(item)
            if ollama_model is not None:
                response = ollama_chat_evaluate(self.config, ollama_model, **item_mapped)
            elif openai_model is not None and openai_client is not None:
                response = openai_client.openai_chat_evaluate(openai_model, **item_mapped)
            else:
                response = self.chat_evaluate_extract(**item_mapped)
            prompt = self.chat_prompt(**item_mapped)
            training_prompt = self.training_prompt(**item_mapped)

            datapoint = OrderedDict()
            datapoint["index"] = i
            datapoint["prompt"] = prompt
            datapoint["training_prompt"] = training_prompt
            if "context" in item_mapped and item_mapped["context"] is not None:
                datapoint["context"] = item_mapped["context"]
            datapoint["question"] = item_mapped["prompt"]
            datapoint["answer"] = item_mapped["response"]
            datapoint["response"] = response
            verify_data.append(datapoint)

        if save_verify_data_path is not None:
            save_json(save_verify_data_path, verify_data)

        errors = 0
        for d in verify_data:
            if ignore_minor(d["response"]) != ignore_minor(d["answer"]):
                errors += 1
                print("Error: answer missmatch")
                print(f"Index: {d['index']}")
                print(f"Answer:\n{d['answer']}")
                print(f"Response:\n{d['response']}")
                print("**********************************************************")
                print()

        print(f"Total mismatches: {errors}")
        return errors

    def test(
        self,
        test_dataset: str,
        save_test_data_path: str | Path | None = None,
        ollama_model: str | None = None,
        openai_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Test model against provided testing dataset.

        This method evaluates the model on the test file.

        Args:
            save_test_data_path (Union[str, Path, None]): Optional path to save
                test results.
            ollama_model (str, optional): Name of Ollama model, if using Ollama instead of app's model

        Returns:
            int: result of the test.

        Example:
            ```python
            app = App()
            test_results = app.test(
                test_dataset="test_data.json", save_test_data_path="test_results.json"
            )
            print(f"Test completed with {len(test_results)} cases")
            ```
        """
        results = []

        if ollama_model is not None and openai_model is not None:
            raise RuntimeError("Both Ollama model and OpenAI model cannot both be set. Only one should be set.")

        openai_client = ChatHPCOpenAI(self.config) if openai_model is not None else None

        test_data = load_json_yaml_arg(test_dataset, False)
        test_data_len = len(test_data)

        for i, item in tqdm(enumerate(test_data), "Test", total=test_data_len):  # type: ignore
            item_mapped = map_keywords(item)
            if ollama_model is not None:
                response = ollama_chat_evaluate(self.config, ollama_model, **item_mapped)
            elif openai_model is not None and openai_client is not None:
                response = openai_client.openai_chat_evaluate(openai_model, **item_mapped)
            else:
                response = self.chat_evaluate_extract(**item_mapped)
            prompt = self.chat_prompt(**item_mapped)
            datapoint = OrderedDict()
            datapoint["index"] = i
            datapoint["prompt"] = prompt
            if "context" in item_mapped and item_mapped["context"] is not None:
                datapoint["context"] = item_mapped["context"]
            datapoint["question"] = item_mapped["prompt"]
            if "response" in item_mapped and item_mapped["response"] is not None:
                datapoint["answer"] = item_mapped["response"]
            datapoint["response"] = response
            results.append(datapoint)

        if save_test_data_path is not None:
            save_json(save_test_data_path, results)

        if "answer" in next(iter(results), {}):  # type: ignore
            errors = 0
            for d in results:
                if ignore_minor(d["response"]) != ignore_minor(d["answer"]):
                    errors += 1
                    print("Missed test:")
                    print(f"Index: {d['index']}")
                    print(f"Answer:\n{d['answer']}")
                    print(f"Response:\n{d['response']}")
                    print("**********************************************************")
                    print()

            correct = test_data_len - errors
            print(f"Total correct: {correct} out of {test_data_len} ({(float(correct)/test_data_len) * 100:.2f}%)")
        return results

    def print_config(self) -> None:
        """Print the current configurations of the application in a formatted table.

        This method displays all configuration settings from self.config in a
        formatted table using the tabulate library.

        The table includes all configuration parameters like:
        - File paths (data, models, checkpoints)
        - Model parameters
        - Training settings
        - Other application settings

        Example:
            ```python
            app = App()
            app.print_config()
            # Outputs:
            # =====================  ==========================
            # Setting               Value
            # =====================  ==========================
            # data_file             /path/to/data.json
            # base_model_path       /path/to/base/model
            # max_response_tokens   600
            # use_wandb            False
            # =====================  ==========================
            ```

        Note:
            The output format uses the 'simple' table format from the tabulate library
            for clean and readable presentation of settings.
        """
        # Get configuration as dict, excluding internal pydantic fields
        config_dict = self.config.model_dump()

        # Format as table rows
        table_data = [[setting, value] for setting, value in config_dict.items()]

        # Define table headers
        headers = ["Setting", "Value"]

        # Print formatted table
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

    def save_readme(self, filename: Path | str) -> None:
        if type(filename) is not Path:
            filename = Path(filename)

        if filename.is_dir():
            filename = filename / "README.md"

        # Get configuration as dict, excluding internal pydantic fields
        config_dict = self.config.model_dump()

        # Replace newlines with newline char.
        config_dict["prompt_template"] = config_dict["prompt_template"].replace("\n", "\\n")

        # Add version
        version_dict = {
            "commit": run("git rev-parse --short HEAD", verbose=False),
            "version": chathpc.app.version,
        }

        # Format as table rows
        table_data = [[setting, value] for setting, value in config_dict.items()]
        version_table_data = [[setting, value] for setting, value in version_dict.items()]

        # Define table headers
        headers = ["Setting", "Value"]

        # Print formatted table
        config_table = tabulate(table_data, headers=headers, tablefmt="github")
        version_table = tabulate(version_table_data, headers=headers, tablefmt="github")

        with open(filename, "w") as fd:
            project_name = Path(run("git rev-parse --show-toplevel", verbose=False)).name.strip()
            fd.write(f"# {project_name} Model Info\n\n## ChatHPC Version Info\n\n")
            fd.write(version_table)
            fd.write("\n\n## Configuration\n\n")
            fd.write(config_table)

    def extract_answer(self, chat_response: str, **kwargs):
        """Extract the model's answer from a complete response string.

        This method processes the full model response to extract just the answer portion,
        removing any template formatting or context that was part of the prompt.

        Args:
            response (str): The complete response string from the model evaluation
            **kwargs: Additional keyword arguments that may be used for template-specific extraction

        Returns:
            str: The extracted answer portion of the response

        Example:
            ```python
            app = App()
            response = app.chat_evaluate("What is Kokkos?")
            answer = app.extract_answer(response)
            print(answer)  # Prints just the model's answer without template formatting
            ```

        Note:
            The exact extraction logic depends on the prompt template structure
            defined in the application configuration.
        """
        chat_answer = chat_response
        chat_answer = chat_answer.replace("<s> ", "").replace("</s>", "")

        prefix = self.inference_template.render(**template_utils.map_keywords(kwargs))
        postfix = self.postfix_template.render(**template_utils.map_keywords(kwargs))
        if chat_answer.startswith(prefix):
            chat_answer = chat_answer[len(prefix) :]

        if chat_answer.endswith(postfix):
            chat_answer = chat_answer[: -len(postfix)]

        return chat_answer
