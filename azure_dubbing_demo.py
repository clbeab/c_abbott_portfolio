import time
import azure.cognitiveservices.speech as speechsdk
import requests, uuid, json

# Speech Service Resource
SPEECH_KEY = 'db22dd2d538c45908dd7914ebc3b58e6'
SPEECH_REGION = 'eastus'

# Translator Resource
key = "774b448f30774e91af79fd76b8c1b0d4"
endpoint = "https://api.cognitive.microsofttranslator.com"
location = 'eastus'

def translation_from_mic():
    # Language Detection
    auto_detect_source_language_config = \
        speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["es-mx", "en-US"])
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_language_detection = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config, auto_detect_source_language_config=auto_detect_source_language_config)
    print('### This would be the caller speaking ###')
    print('Language detection... say something please.')
    result = speech_language_detection.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        detected_src_lang = result.properties[
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult]
        print("Detected Language: {}".format(detected_src_lang))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

    if detected_src_lang == 'es-mx':
        detected_text = 'spanish'
        source_lang = 'es-mx'
        alt_src = 'es'
        translated_lang = 'en-US'
    else:
        detected_text = 'english'
        source_lang = 'en-US'
        alt_src = 'en'
        translated_lang = 'es-mx'

    # Translate standard response to source language
    params = {
    'api-version': '3.0',
    'from': 'en',
    'to': source_lang
    }
    headers = {
    'Ocp-Apim-Subscription-Key': key,
    'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
    }

    path = '/translate'
    constructed_url = endpoint + path
    body = [{
        'text': f'We detected you are speaking {detected_text}.  Operator responses will be automatically translated into {detected_text} for you. Please wait for the translated responses.'
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    print("\n### Standard response in caller's language ###")
    standard_text_in_src_lang = ((response[0]['translations'][0]['text']))
    print(f'English: we detected you are speaking {detected_text}.  Operator responses will be automatically translated into {detected_text} for you. Please wait for the translated responses.')
    print(f'Translated: {standard_text_in_src_lang}\n')

    # Translated Speaker
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_language = source_lang
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    speech_synthesizer.speak_text_async(standard_text_in_src_lang).get()

    # Translation and Transcription
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=SPEECH_KEY, region=SPEECH_REGION,
        speech_recognition_language=translated_lang,
        target_languages=('en', 'es'))
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config, audio_config=audio_config)

    # Event handler for translator/transcriber.  Calls the translated speaker after translating.
    def result_callback(event_type: str, evt: speechsdk.translation.TranslationRecognitionEventArgs):
        """callback to display a translation result"""
        print("English: {}\nSpanish: {}\n----------".format(
            evt.result.translations['en'],evt.result.translations['es']))
        result = speech_synthesizer.speak_text_async(evt.result.translations[alt_src]).get()
        
    def canceled_cb(evt: speechsdk.translation.TranslationRecognitionCanceledEventArgs):
        print('CANCELED:\n\tReason:{}\n'.format(evt.result.reason))
        print('\tDetails: {} ({})'.format(evt, evt.result.cancellation_details.error_details))
    
    done = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Events
    recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED: {}'.format(evt)))
    recognizer.recognized.connect(lambda evt: result_callback('RECOGNIZED', evt))
    recognizer.canceled.connect(canceled_cb)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)

    def synthesis_callback(evt: speechsdk.translation.TranslationRecognitionEventArgs):
            """
            callback for the synthesis event
            """
            print('SYNTHESIZING {}\n\treceived {} bytes of audio. Reason: {}'.format(
                evt, len(evt.result.audio), evt.result.reason))

    recognizer.synthesizing.connect(synthesis_callback)

    # start translation
    print('### Now speaking as operator ###')
    print('-- type "stop" to exit --\n')
    recognizer.start_continuous_recognition()

    while not done:
        # Continuous mode exited by typing 'stop'
        stop = input()
        if (stop.lower() == "stop"):
            done = True
        time.sleep(.5)

    recognizer.stop_continuous_recognition()

if __name__ == '__main__':
    translation_from_mic()