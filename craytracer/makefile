# Compiler and flags
CC = gcc
CFLAGS = -fopenmp -I ./include
LDFLAGS = -lm

# Source files
SRCS = main.c \
       sources/allocator.c \
       sources/camera.c \
       sources/color.c \
       sources/hitRecord.c \
       sources/material.c \
       sources/outfile.c \
       sources/ray.c \
       sources/sphere.c \
       sources/texture.c \
       sources/util.c

# Output binary
TARGET = raytracer

# Build rule
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGET)
