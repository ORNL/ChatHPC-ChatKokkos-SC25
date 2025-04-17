# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to a custom Development Versioning specified by Aaron Young.

A summary of Development Versioning Specification is shown below.

> Given a version number BRANCH.TAG.BUILD, increment the:
> 1. BRANCH version when you make breaking/major changes that you want to track in a separate branch.
> 2. TAG version when you make a new tag to mark a specific spot.
> 3. BUILD version when you create a new build with artifacts or bug fixes for that you want to point to.
>
> Then for your repo you have branch versions for each version. For example branches v0 and v1. Then when you create tags, say on branch v0, you would create tags v0.0.0, v0.1.0, and v0.2.0.
> CI or a manual process could add v0.0.x branches as new changes are added to a local branch. BUILD is also used when patches are applied to a tagged branch, after the patch is applied, add a new tag with BUILD + 1.
>
> `main` always points to the current major branch plus 1. `dev` is an integration branch before merging into `main`. When `dev` is merged into `main`, the TAG is updated.

An alternative approach is to use date-based versioning.

With this method, the version is YEAR.MONTH.RELEASE. To increment this version, use the year and the date without 0 padding for the first two numbers. I prefer to use the year without the centary. Then increment the RELEASE number to a unique release. This process is done automatically by the `scripts/version_bump.py` script. Using this script is the prefered method for versioning without planned backporting of fixes.

## [Unreleased]

## [0.0.2] - 2025-04-17

Various revisions to get ready for the initial submission.

## [0.0.1] - 2025-04-15

Initial release of paper artifacts.

[unreleased]: https://github.com/ORNL/ChatHPC-ChatKokkos-SC25/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/ORNL/ChatHPC-ChatKokkos-SC25/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ORNL/ChatHPC-ChatKokkos-SC25/releases/tag/v0.0.1
