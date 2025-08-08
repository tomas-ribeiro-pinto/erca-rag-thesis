class RagUser:
    def __init__(self, user_id: str):
        self.user_id = user_id

    def get_history(self):
        """Retrieve the user's interaction history."""
        # This method should interact with a database or storage to get the user's history
        # For now, we will return a placeholder
        return f"History for user {self.user_id}"