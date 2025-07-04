# MusicGen Audio Generator

A simple Gradio web application that generates music from text descriptions using Meta's MusicGen model.

## Features

- Generate music from text descriptions
- Unconditional music generation option
- GPU acceleration support
- User-friendly interface with example prompts

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/musicgen-gradio.git
cd musicgen-gradio
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python app.py
```

2. Open your web browser and navigate to the provided URL (typically http://127.0.0.1:7860)

3. Enter a description of the music you want to generate in the text box and click "Generate Music"

## Example Prompts

- "An electronic dance track with a heavy beat and synthesizer melody"
- "A peaceful piano sonata with gentle flowing notes"
- "An upbeat jazz piece with saxophone and trumpet solos"
- "A rock anthem with electric guitar riffs and powerful drums"

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)

## Notes

- The first generation may take longer as the model loads into memory
- Generation times vary based on your hardware (10-30 seconds typically)
- For best results, provide detailed descriptions including style, mood, instruments, and tempo
