# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
---------------------------------------------------------

## [Unreleased]

---------------------------------------------------------
 ## [0.4.0] - 2024-06-17

### Fixed

* updated some imports for pydantic and `molflux==0.5.0`

 ## [0.3.1] - 2024-05-31

### Fixed

* check pbc is True

 ## [0.3.0] - 2024-04-25

### Added

* Added graph attrs in graph datasets/loaders

## [0.2.4] - 2024-04-24

### Fixed

* Added coordinate wrapping in the neighbour list computations of ANI and graphs

## [0.2.3] - 2024-04-09

## Changed

* Remove torch < 2.1 pin

## [0.2.2] - 2024-04-05

## Fixed

* Fixed problem with openmm precision (force cast positions to model dtype)

## [0.2.1] - 2024-04-03

## Fixed

* Fixed bug in neighbour list with torchscript

## [0.2.0] - 2024-03-07

## Fixed

* Improved neighbour list computation (10x faster)

## [0.1.1] - 2024-02-20

## Fixed

* fixed import names for `torch_geometric>=2.5.0`

## [0.1.0] - 2024-02-19

## Added

* Added tracking for individual loss components

## [0.0.2] - 2024-02-09

## Fixed

* Fixed EGNN pooling bug

## [0.0.1] - 2023-12-15

## Fixed

* Fixed featuriser

## [0.0.0] - 2023-12-15

## Added

* Initial release

---------------------------------------------------------

## [X.Y.Z] - 20YY-MM-DD

(Template)

### Added

For new features.

### Changed

For changes in existing functionality.

### Deprecated

For soon-to-be removed features.

### Removed

For now removed features.

### Fixed

For any bug fixes.

### Security

In case of vulnerabilities.
