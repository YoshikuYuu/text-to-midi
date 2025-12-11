#Text-2-Midi-Mini

## Repository Structure and Dependencies

This repository is organized into two main components:

- **`mini_midi/`** – contains the model implementation, training code, and configuration files for Minitext2midi.
- **`mini_midi_eval/`** – contains the evaluation scripts used to compute all metrics reported in the paper.

### External Requirements

The evaluation pipeline requires access to the following resources:

1. **Trained Minitext2midi model weights**  
2. **The original text2midi model checkpoint** (available from the official text2midi GitHub repository)  
3. **GigaMIDI** and **MidiCaps** datasets for evaluation and comparison  

Due to their size, these files could not be included in this repository, as GitHub enforces a 50MB file limit. Users who wish to reproduce our experiments must download these assets separately and update the relevant paths in the evaluation configuration files.

### Note

Once the datasets and model checkpoints are properly configured, the scripts in `mini_midi_eval/` can be used to reproduce all evaluation metrics reported in our paper.
