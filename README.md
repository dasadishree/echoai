# Echo AI
Developing AI tool that greets people with intellectual disabilitities / neurodivergent people and recognizes people's voices / can differentiate between different people.

HOW TO RUN:
- Training data located in the labeled_samples folder
- can download more data from youtube using the get_samples.py (python3 get_samples.py)
- Start by training model w existing audio, this generates speaker profiles (python train_speaker_model.py --data_dir labeled_samples --output speaker_profiles.pkl)
- test model using new clip by doing (python recognize_speaker.py test_voice.wav --profiles speaker_profiles.pkl) if the mystery clip is named test_voice.wav
-- use python recognize_speaker.py labeled_samples/JayZ.wav to see just the greeting
-- use python recognize_speaker.py labeled_samples/JayZ.wav --info to see technical breakdown (scores and confidence as well as greeting)
- if u want to add new training data, just re-run train_speaker_model.py scipt and new voices r added to the profile database (speaker_profiles.pkl)
- results explanation: identified: gives u a profile and a confideence level, then combines cosine similarity of every profile saved. (1.0 = exact match (max), close to 1 is more similar, so smaller negative numbers mean they sound less alike)

NOTES:
1/29/26:
Objective: Greet people w intellectual disability by recognizing their voice
Target: Neurodiverse kids with learning disorders like autism, ADHD, language development disabilities, etc
- Train AI to identify each person's voice (training with my own voice first) and recognize where the voice is coming from
- Try Antigravity
- ChatGPT (generative) CursorAI (agentic) vs Antigravity (AGI)
- Learn difference between these types of AI
- AGI (artificial generative intelligence) automates

2/5/26:
1. Download entire code into local computer (from Github repository)  (done)
2. let CursorAI/Claude AI to run the code (done)
3. create more than 50 audio data, from celebrities online, myself, & family members (done)
4. train your model with this audio data (done)
5. Validate. it should be 100% accuracy if validating with the existing 50 audio data (done)
6. app should be able to greet the person based on the audio. ex: if its audio of my dad it should say/show hello dad (done)
7. add 20 more audio files that havent been used as training data and validate with these to see if the model really works or not, record accuracy

- Use API input: user voice training maybe saying a certain snentnece for paragraph
output: api returns the name of the speaker
- use fast api, reinformenet learning? 

TO DO:
- read through wespeaker documentation: https://wenet-e2e.github.io/wespeaker/
- check out antigravity?
-Docker image runs linux ubuntu inside docker
- docker desktop adding it to
- ubuntu 20.4
- this makes it easier to deploy later on and donwloading dependices becomes easier too
- run the github code from microsoft teams 