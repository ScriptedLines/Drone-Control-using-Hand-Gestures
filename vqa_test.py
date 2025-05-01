import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")
# Step 2: Load the image and preprocess it
image_path = "C:\\Users\\Aaryaman Bisht\\Downloads\\download.png"  # Path to your image
image = Image.open(image_path).convert("RGB")

# Step 3: Define the question
question = "Which flag is there in this image?"  # Replace with your question

# Step 4: Process the image and question using the processor
inputs = processor(images=image, text=question, return_tensors="pt")

# Step 5: Perform inference (answer prediction)
with torch.no_grad():
    outputs = model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)

# Step 6: Print the predicted answer
print(f"Predicted Answer: {answer}")
