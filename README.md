# Effective Cell Selection Strategies in Wave Function Collapse for Minecraft

## Project Overview
This repository contains the implementation of the research study **"Determining Effective Cell Selection Strategies in Wave Function Collapse for Minecraft"**, conducted by Shailyn Ramsamy Moodley. The study investigates how different cell selection strategies impact the architectural quality of procedurally generated Minecraft buildings using the Wave Function Collapse (WFC) algorithm.

### Research Highlights
- **Four Cell Selection Strategies Evaluated**:
  - Entropy-Based Selection
  - Height Priority
  - Center-Based Growth
  - Random Walk
- **Metrics Used**:
  - Pattern Diversity
  - Structural Connectivity
  - Vertical Coherence
  - Pattern Distribution Regularity
- **Key Findings**:
  - Each strategy has unique strengths tailored to different architectural goals.
  - Entropy-based strategies provide balance and adaptability.
  - Height-priority strategies ensure structural stability.
  - Center-based approaches create symmetrical designs.
  - Random-walk approaches generate organic and naturalistic structures.

## Prerequisites
- Python 3.8 or later
- Minecraft Java Edition
- [GDMC HTTP Interface Mod](https://github.com/Niels-NTG/gdmc_http_interface)
- [GDPC Library](https://github.com/Niels-NTG/gdpc)
- Required Python packages:
  - `numpy`
  - `matplotlib`
  - `scipy`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Shailyn-Ramsamy/Cell-Selection-Strategies-for-Minecraft-WFC.git
## How to Run
1. Ensure you have **Minecraft Java Edition** installed with the [GDMC HTTP mod](https://github.com/Niels-NTG/gdmc).
2. Launch the modded Minecraft server and verify that the GDMC HTTP interface is running.
3. CD to project directory in command prompt.
4. Run the `build_house` script:
   ```bash
   python build_house.py
5. Input strategy, position, and dimensions

![Wave Function Collapse Demo](wfcdemo.gif)
