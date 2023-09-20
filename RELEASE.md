# Release Notes

The codebase follows [Semantic Versioning](https://semver.org/); the convention
for most software products. In summary, this means the version numbers should be read in the
following way.

Given a version number MAJOR.MINOR.PATCH (e.g. 1.0.0), increment the:

- MAJOR version when you make incompatible API changes,
- MINOR version when you add functionality in a backwards compatible manner, and
- PATCH version when you make backwards compatible bug fixes.

Note that the main branch of this repository contains a work in progress, and  may **not**
contain a stable version of the codebase. We aim to keep the master branch stable, but for the
most stable versions, please see the
[releases](https://github.com/Transport-for-the-North/caf.distribute/releases)
page on GitHub. A log of all patches made between versions can also be found
there.

Below, a brief summary of patches made since the previous version can be found.

### Next Release Notes
- Refactored the gravity model, the class now has 4 main methods
  - `run()` - Run the gravity model with the given cost params etc.
  - `run_with_perceived_factors()` - Builds on `run()` If the target is not reached, apply perceived factors to improve the run.
  - `calibrate()` - Calibrate the gravity model cost parameters to provide the best possible fit to a given cost distribution.
  - `calibrate_with_perceived_factors()` - Builds on `calibrate()` If the target is not reached, apply perceived factors and calibrate some more.
- The results of a GravityModel run are now stored in a  `GravityModelResults` class.
