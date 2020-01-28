# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Get Started!
Ready to contribute? Here's how to set up `fnet` for local development.

* Fork the `fnet` repo on GitHub.
* Clone your fork locally:

```
$ git clone --recurse-submodules git@github.com:{your_name_here}/fnet.git
```

* Install the project in editable mode. (It is also recommended to work in a virtualenv or anaconda environment):

```
$ cd fnet/
$ pip install -e .[dev]
```

* Create a branch for local development:

```
$ git checkout -b {your_development_type}/short-description
```
Ex: feature/read-tiff-files or bugfix/handle-file-not-found<br>
Now you can make your changes locally.<br>

* When you're done making changes, check that your changes pass linting and tests, including testing other Python
versions with make:

```
$ make build
```

* Commit your changes and push your branch to GitHub:

```
$ git add .
$ git commit -m "Resolves gh-###. Your detailed description of your changes."
$ git push origin {your_development_type}/short-description
```

* Submit a pull request through the GitHub website.

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

```
$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags
```

Make and merge a PR to branch `stable` and GitHub will then deploy to PyPI once merged.
