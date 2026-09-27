# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest (main branch) | ✅ |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please **do not open a public GitHub issue**.

Instead, report it privately by emailing:

**anjanatiha@gmail.com**

Please include:
- A description of the vulnerability
- Steps to reproduce it
- Any potential impact

I will respond within 48 hours and work to resolve confirmed vulnerabilities promptly.

## Scope

This project runs fully locally — no cloud services, no API keys, no user data is transmitted externally. The primary security considerations are:

- Malicious document uploads (PDF, DOCX, etc.)
- Unsafe content in web URLs fetched for indexing
- Local file system access via document loaders

## Thank You

Security researchers who responsibly disclose vulnerabilities will be credited in the project changelog.
