class VoiceCommandProcessor:
    def __init__(self):
        self.commands = {
            'status': self.get_status,
            'position': self.get_position,
            'execute': self.execute_trade
        }
    def process_command(self, audio_input):
        command = self.recognize_speech(audio_input)
        if command in self.commands:
            return self.commands[command]()
