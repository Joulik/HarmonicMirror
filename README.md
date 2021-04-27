# HarmonicMirror

## Principle

Construction of the image captured by webcam starting from a random spatial distribution of pixels. The final image is constructed from edges detected in the webcam snapshot. See an example below. 

![anim](https://github.com/Joulik/Joulik.github.io/blob/master/images/fun_python.gif)

## Usage

```
# Start the script in python
python harmonic-mirror.py
```

Pay attention there are three important parameters that the user may want to change by editing the script.

- max_number_captures, which is the number of times the webcam captures an image.

- produce_gif, which when set to True yields the script to produce an animated gif of the captures constructions. Pay attention it may generate really large files. For example, 4 captures yield a gif file of about 50MB.

- reservoir_factor, which must be decreased if the program stops with IndexError message.

## TODO

Make usage more user friendly.

## Requirements

- PIL

- opencv

- imageio

- numpy