from marshmallow import Schema, fields

class ChatbotOutput(object):
    def __init__(self, output):
        self.output = output

    def __repr__(self):
        return '<ChatbotOutput(output={self.output!r})>'.format(self=self)

class ChatbotOutputSchema(Schema):
    output = fields.Str()