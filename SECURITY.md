# Security

## Reporting a problem

Please do not open a public issue. Use **Report a vulnerability** in the Security tab of this repository.

This is a personal project: fixes come on a best-effort basis.

## What to know before running Vigie

- **There is no authentication.** Anyone who reaches the port can start scans and delete them. Keep it on `127.0.0.1`, or behind an SSH tunnel or a private network.
- **The server refuses unknown host names and cross-origin writes**, to protect it from web pages open in your browser. This does not replace authentication.
- **Weights are code.** A `.pt` file can run code when loaded. `vigie fetch-model` pins a revision and checks a SHA-256. With `VIGIE_MODEL`, the trust is yours.
- **Hosted vision models see your data.** With a remote `VIGIE_LLM_URL`, ship crops and scan data leave your machine.
- **Keys stay out of the repo.** Use the environment or `~/.config/vigie/aisstream.key`. `.env` and `*.key` files are ignored by git.
