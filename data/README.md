# Data Directory

This directory contains the image-caption dataset for training and evaluation.

## Structure

```
data/
├── images/           # Directory containing all images
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── captions.json     # Main captions file
├── captions_clean.json    # Cleaned captions
└── captions_upgraded.json # Enhanced captions
```

## Caption Format

The `captions.json` file should be a JSON array with the following structure:

```json
[
  {
    "file_name": "image_1.jpg",
    "caption": "white chair on dark hardwood floor"
  },
  {
    "file_name": "image_1.jpg",
    "caption": "dining chair on dark hardwood floor"
  },
  {
    "file_name": "image_2.jpg",
    "caption": "mug on white chair on dark hardwood floor"
  }
]
```

## Notes

- Each image can have multiple captions
- Captions should be lowercase and descriptive
- Image file names must match the entries in captions.json
- The `images/` subdirectory must contain the actual image files
