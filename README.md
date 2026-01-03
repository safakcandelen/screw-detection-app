# Vida Asistanı (Screw Head Detection)

This is a Streamlit application compliant with the requested specifications for detecting screw heads and recommending the correct tool bit.

## Installation

1.  Make sure you have Python installed.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

1.  Ensure your trained model file `best.pt` is located in this root directory.
2.  Run the Streamlit server:

```bash
streamlit run app.py
```

3.  The application will open in your default web browser.

## Features

-   **Model**: Uses YOLOv8 (`best.pt`) for detection.
-   **Inputs**: Supports both Camera Input and File Upload.
-   **Recommendations**: Provides specific tool recommendations and warnings for Tort (T), Phillips (PH), Pozidriv (PZ), Hex (H), and Slotted (SL) heads.
