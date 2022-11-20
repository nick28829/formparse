============
Contributing
============

Welcome to ``formparse`` contributor's guide.

This document focuses on getting any potential contributor familiarized
with the development processes, but `other kinds of contributions`_ are also
appreciated.

If you are new to using git_ or have never collaborated in a project previously,
please have a look at `contribution-guide.org`_. Other resources are also
listed in the excellent `guide created by FreeCodeCamp`_.

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt, `Python Software
Foundation's Code of Conduct`_ is a good reference in terms of behavior
guidelines.


Issue Reports
=============

If you experience bugs or general issues with ``formparse``, please have a look
on the `issue tracker`_. If you don't see anything useful there, please feel
free to fire an issue report.

.. tip::
   Please don't forget to include the closed issues in your search.
   Sometimes a solution was already reported, and the problem is considered
   **solved**.

New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.


Documentation Improvements
==========================

You can help improve ``formparse`` docs by making them more readable and coherent, or
by adding missing information and correcting mistakes.

``formparse`` documentation uses Sphinx_ as its main documentation compiler.
This means that the docs are kept in the same repository as the project code, and
that any documentation update is done in the same way was a code contribution. We
are using reStructuredText_ for our documents.

   .. tip::
      Please notice that the `GitHub web interface`_ provides a quick way of
      propose changes in ``formparse``'s files. While this mechanism can
      be tricky for normal code contributions, it works perfectly fine for
      contributing to the docs, and can be quite handy.

      If you are interested in trying this method out, please navigate to
      the ``docs`` folder in the source repository_, find which file you
      would like to propose changes and click in the little pencil icon at the
      top, to open `GitHub's code editor`_. Once you finish editing the file,
      please write a message in the form at the bottom of the page describing
      which changes have you made and what are the motivations behind them and
      submit your proposal.

When working on documentation changes in your local machine, you can
compile them using Sphinx_::

    sphinx-build docs docs/_build

and use Python's built-in web server for a preview in your web browser
(``http://localhost:8000``)::

    python3 -m http.server --directory 'docs/_build/html'


Code Contributions
==================

``formparse`` is built using Pythons ``ast`` module which parses the input of a
formula as if it were Python code and then only enables the operations that we
want to support and that are safe to use.

Submit an issue
---------------

For every submission that is not trivial, e.g. a simple typo, we'd like you to create
an issue in our `issue tracker`_. At the begining of the PR name, please add either
``feat:``, ``fix:`` or ``chore``, depending if it is a new feature, a bug fix or a
chore. 

Create an environment
---------------------

Before you start coding, we recommend creating an isolated virtual
environment to avoid any problems with your installed Python packages.
This can easily be done via ``venv``::

    python -m venv <PATH TO VENV>
    source <PATH TO VENV>/bin/activate

Clone the repository
--------------------

#. Create an user account on |the repository service| if you do not already have one.
#. Fork the project repository_: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on |the repository service|.
#. Clone this copy to your local disk::

    git clone git@github.com:YourLogin/formparse.git
    cd formparse

#. You should run::

    pip install -U pip setuptools -e .

   to be able to import the package under development in the Python REPL.

Implement your changes
----------------------

#. Create an issues or reference an existing one. We think that every PR should
   be matched to an issue. This allows discussion and avoids that two people work
   on the same feature without their knowledge. Check also the section about
   creating issues.

#. Create a branch to hold your changes::

    git checkout -b <issue-number>-my-feature

   and start making changes. Never work on the main branch!

#. Start your work on this branch. Don't forget to add docstrings_ to new
   functions, modules and classes, especially if they are part of public APIs.

#. Add yourself to the list of contributors in ``AUTHORS.rst``.

#. When you’re done editing, do::

    git add <MODIFIED FILES>
    git commit


   to record your changes in git_. Please start every commit message with 
   ``#<issue-number>`` to ensure we can match your commits to a issue after
   merging (in case we don't squash it).


   .. important:: Don't forget to add unit tests and documentation in case your
      contribution adds an additional feature and is not just a bugfix.

      Moreover, writing a `descriptive commit message`_ is highly recommended.
      In case of doubt, you can check the commit history with::

         git log --graph --decorate --pretty=oneline --abbrev-commit --all

      to look for recurring communication patterns.

#. Please check that your changes don't break any unit tests with::

    pip install -e . && python -m unittest discover tests

Submit your contribution
------------------------

#. If everything works fine, push your local branch to |the repository service| with::

    git push -u origin my-feature

#. Go to the web page of your fork and click |contribute button|
   to send your changes for review.

Find more detailed information in `creating a PR`_. You might also want to open
the PR as a draft first and mark it as ready for review after the feedbacks
from the continuous integration (CI) system or any required fixes.


Troubleshooting
---------------

The following tips can be used when facing problems to build or test the
package:

#. Make sure to fetch all the tags from the upstream repository_.
   The command ``git describe --abbrev=0 --tags`` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   You can also try to remove all the egg files or the complete egg folder, i.e.,
   ``.eggs``, as well as the ``*.egg-info`` folders in the ``src`` folder or
   potentially in the root of your project.


Maintainer tasks
================

Releases
--------

If you are part of the group of maintainers and have correct user permissions
on PyPI_, the following steps can be used to release a new version for
``formparse``:

#. Make sure all unit tests are successful.
#. Tag the current commit on the main branch with a release tag, e.g., ``v1.2.3``.
#. Push the new tag to the upstream repository_, e.g., ``git push upstream v1.2.3``
#. Clean up the ``dist`` and ``build`` folders with ``rm -rf dist build``
   to avoid confusion with old builds and Sphinx docs.
#. Run ``python -m build`` and check that the files in ``dist`` have
   the correct version (no ``.dirty`` or git_ hash) according to the git_ tag.
   Also check the sizes of the distributions, if they are too big (e.g., >
   500KB), unwanted clutter may have been accidentally included.
#. Run ``twine check dist/*`` to check if there are any errors detected.
#. Upload it as a test to test-PyPI with twine upload -r testpypi dist/*
#. Run ``twine upload dist/*`` and check that everything was
   uploaded to PyPI_ correctly.


.. |the repository service| replace:: GitHub
.. |contribute button| replace:: "Create pull request"

.. _repository: https://github.com/nick28829/formparse
.. _issue tracker: https://github.com/nick28829/formparse/issues

.. |virtualenv| replace:: ``virtualenv``
.. |pre-commit| replace:: ``pre-commit``

.. _black: https://pypi.org/project/black/
.. _CommonMark: https://commonmark.org/
.. _contribution-guide.org: https://www.contribution-guide.org/
.. _creating a PR: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
.. _descriptive commit message: https://chris.beams.io/posts/git-commit
.. _docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _first-contributions tutorial: https://github.com/firstcontributions/first-contributions
.. _flake8: https://flake8.pycqa.org/en/stable/
.. _git: https://git-scm.com
.. _guide created by FreeCodeCamp: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _MyST: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
.. _other kinds of contributions: https://opensource.guide/how-to-contribute
.. _pre-commit: https://pre-commit.com/
.. _PyPI: https://pypi.org/
.. _PyScaffold's contributor's guide: https://pyscaffold.org/en/stable/contributing.html
.. _Pytest can drop you: https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _tox: https://tox.wiki/en/stable/
.. _virtual environment: https://realpython.com/python-virtual-environments-a-primer/
.. _virtualenv: https://virtualenv.pypa.io/en/stable/

.. _GitHub web interface: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
.. _GitHub's code editor: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
