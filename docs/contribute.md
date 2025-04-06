# Contributing to Fingerprint Recognition System

Thank you for your interest in contributing to the Fingerprint Recognition System! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/your-username/fingerprint-pro.git
cd fingerprint-pro
```

3. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

1. Make your changes following the project's coding standards
2. Write or update tests for your changes
3. Run the test suite:
```bash
cd backend/python
python -m pytest tests/
```

4. Update documentation if necessary
5. Commit your changes with a descriptive commit message
6. Push your branch to your fork
7. Create a Pull Request (PR)

## Coding Standards

### Python Code

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable and function names

### Documentation

- Keep documentation up-to-date
- Use clear and concise language
- Include examples where helpful
- Document all public APIs

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting a PR
- Include edge cases in your tests
- Maintain or improve test coverage

## Pull Request Process

1. Ensure your PR description clearly describes the problem and solution
2. Include relevant issue numbers if applicable
3. Update the CHANGELOG.md if your changes are significant
4. Wait for review and address any feedback

## Review Process

- PRs will be reviewed by maintainers
- Reviews may request changes or improvements
- Be responsive to review comments
- Keep PRs focused and manageable in size

## Reporting Issues

When reporting issues, please include:

1. A clear description of the problem
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, etc.)
6. Relevant error messages or logs

## Feature Requests

For feature requests:

1. Check if the feature has already been requested
2. Provide a clear description of the feature
3. Explain the use case and benefits
4. Suggest implementation details if possible

## Questions and Support

- Check the documentation first
- Search existing issues
- If your question isn't answered, open a new issue
- Be specific about your question and provide context

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE) file. 