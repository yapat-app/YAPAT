# YAPAT — Yet Another PAM Annotation Tool

YAPAT is an interactive annotation tool that uses machine learning to prioritize high-value samples for labeling, reducing the annotation effort required for training data collection.

## Repository Structure

This repository links the core components of YAPAT as Git submodules:

| Directory    | Repository          | Description                          |
|-------------|---------------------|--------------------------------------|
| `frontend/`  | [yapat-frontend](https://github.com/yapat-app/yapat-frontend) | TypeScript/React annotation interface |
| `backend/`   | [yapat-backend](https://github.com/yapat-app/yapat-backend)   | Python backend and ML pipeline        |
| `oe_yapat/`  | [oe_yapat](https://github.com/yapat-app/oe_yapat)             | Ontology integration for YAPAT        |
| `WSSED/`     | [WSSED-YAPAT](https://github.com/MammadliN/WSSED-YAPAT)      | WebSocket server for real-time communication |

## Getting Started

Clone this repository with all submodules:

```bash
git clone --recurse-submodules https://github.com/yapat-app/yapat.git
```

Or, if you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Refer to the README in each submodule for setup and usage instructions.

## License

YAPAT source code is available under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
