from marshmallow import Schema, fields

class UserPrompt(object):
    def __init__(self, prompt):
        self.prompt = prompt

    def __repr__(self):
        return '<UserPrompt(prompt={self.prompt!r})>'.format(self=self)

class UserPromptSchema(Schema):
    prompt = fields.Str()