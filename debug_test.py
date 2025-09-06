import google.generativeai as genai

# 🔑 Put your Gemini API key directly here
genai.configure(api_key="AIzaSyAkQ0Y0ppBvu2_Phd42Jq8gq0CnhaL3SWE")

# Initialize model
model = genai.GenerativeModel("gemini-1.5-flash")

# Few-shot prompt with examples
few_shot_prompt = """
You are a Fake News Detector.
Your task: Classify text as either FAKE or REAL, then give a short explanation.

Follow these rules:
- Always reply with "FAKE" or "REAL" as the first word.
- Then provide 1-2 sentences of reasoning.

Example 1:
Input: "NASA confirms aliens landed in Delhi today!"
Output: FAKE - NASA has made no such announcement and major media outlets have not reported this.

Example 2:
Input: "The WHO announced a new flu vaccine update."
Output: REAL - WHO regularly releases such health updates.

Example 3:
Input: "XYZ claims chocolate cures cancer."
Output: FAKE - There is no scientific evidence that chocolate can cure cancer.

Now classify this:
Input: "Apple launches new iPhone model with improved camera."
Output:
"""

# Call Gemini with the few-shot prompt
resp = model.generate_content(few_shot_prompt)

print("Gemini Response:\n", resp.text)
