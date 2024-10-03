#!/bin/bash
cd results
ffmpeg -framerate 5 -i ray_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4