import torch
import numpy as np
import scipy.io.wavfile
import gradio as gr
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import os

def generate_music(prompt, use_gpu=True, unconditional=False):
    # Check if GPU is available and should be used
    device = "cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = AutoModelForTextToWaveform.from_pretrained("facebook/musicgen-small")
    model.to(device)
    
    # Generate music
    if unconditional:
        print("Generating unconditional music...")
        unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
        audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
    else:
        print(f"Generating music with prompt: '{prompt}'")
        processor = AutoTokenizer.from_pretrained("facebook/musicgen-small")
        inputs = processor(
            text=prompt,
            padding=True,
            return_tensors="pt",
        )
        audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # Ensure audio_values is 1D and scale if necessary
    audio_data = audio_values[0].cpu().numpy()
    
    # Check if audio_data is in the correct format
    if audio_data.ndim > 1:
        audio_data = audio_data[0]  # Take the first channel if stereo

    # Audio data is already in the range [-1, 1] for gradio
    return (sampling_rate, audio_data)

# Define Gradio interface
def create_gradio_app():
    with gr.Blocks(title="MusicGen Audio Generator") as app:
        gr.Markdown("# MusicGen Audio Generator")
        gr.Markdown("Generate music from text prompts using Meta's MusicGen model.")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Music Description", 
                    placeholder="Enter a description of the music you want to generate...",
                    lines=3
                )
                
                with gr.Row():
                    use_gpu = gr.Checkbox(label="Use GPU (if available)", value=True)
                    unconditional = gr.Checkbox(label="Unconditional Generation (ignores prompt)")
                
                generate_btn = gr.Button("Generate Music", variant="primary")
                
            with gr.Column():
                output_audio = gr.Audio(label="Generated Music", type="numpy")
                
        generate_btn.click(
            fn=generate_music,
            inputs=[prompt, use_gpu, unconditional],
            outputs=output_audio
        )
        
        gr.Markdown("## Example Prompts")
        example_prompts = [
            "An electronic dance track with a heavy beat and synthesizer melody",
            "A peaceful piano sonata with gentle flowing notes",
            "An upbeat jazz piece with saxophone and trumpet solos",
            "A rock anthem with electric guitar riffs and powerful drums"
        ]
        
        gr.Examples(
            examples=[[p] for p in example_prompts],
            inputs=prompt
        )
        
        gr.Markdown("### Notes")
        gr.Markdown("- First generation may take longer as the model loads")
        gr.Markdown("- Generation may take 10-30 seconds depending on your hardware")
        gr.Markdown("- For best results, provide detailed descriptions of style, mood, instruments, and tempo")
        
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True)