# Security Policy

## Supported versions

The latest released version receives security fixes.

## Reporting a vulnerability

Please do **not** open a public issue for security vulnerabilities.

Instead, report them privately by emailing **1dg618@gmail.com**, or by using
GitHub's [private vulnerability reporting](https://github.com/1dg618/fakellm-recorder/security/advisories/new).

Include a description of the issue, steps to reproduce, and the affected version.
You can expect an acknowledgement within a few days.

## Note on credentials

fakellm-recorder records real API traffic. Recorded sessions and generated
config files may contain sensitive prompt data. Credentials are stripped at
capture and PII scrubbing is on by default, but always review generated files
before committing them.
