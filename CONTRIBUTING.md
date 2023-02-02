# **Contribution Guidelines**

Contributions are welcome and appreciated, however before submiting a pull request make sure to read this document and that the pull request follows all of the guidelines.  
We also reserve the right to not accept contributions that add no meaningful changes or go out of the scope of the project.

## **Contributions:**

### 🐛 **1. Report Bugs**

Report bugs at <https://github.com/HenriquesLab/NanoPyx/issues>.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### 🐞 **2. Fix Bugs**

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

### 🗣️ **3. Submit Feedback**

The best way to send feedback is to file an issue at <https://github.com/HenriquesLab/NanoPyx/issues>.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.

## **Get Started!**

Ready to contribute? Here's how to set up **nanopyx** for local development.

1. Fork the **nanopyx** repo on GitHub.
2. Clone your fork locally::

   `$ git clone git@github.com:your_name_here/nanopyx.git`

3. Follow the installation instructions in the readme to install a nanopyx environment.

4. Create a branch for local development::

   `$ git checkout -b name-of-your-bugfix-or-feature`

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass our tests::

   `$ pytest`

   Make sure you have installed the optional nanopyx testing dependencies.

6. Commit your changes and push your branch to GitHub::

   `$ git add .`  
   `$ git commit -m "Your detailed description of your changes."`  
   `$ git push origin name-of-your-bugfix-or-feature`

7. Submit a pull request through the GitHub website.

## **Pull Request Guidelines**

Before you submit a pull request, check that it meets these guidelines:

- The pull request should pass every existing test.
- The pull request should work for Python 3.8, 3.9 and 3.10
- If the pull request adds functionality, new automated tests should be included in the pull request covering the new functionality
- Please ensure pytest coverage at least stays the same before you submit a pull request.
