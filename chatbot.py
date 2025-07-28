from transformers import pipeline

# Load a lightweight QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define your health-related chatbot function
def get_bot_response(question):
    context = """
    Shaking Palsy, also known as Parkinson’s disease, is a neurodegenerative disorder that affects movement.
    Common symptoms include tremors, stiffness, slowness of movement, and difficulty with balance and coordination.
    There is no cure, but early diagnosis and medication can help manage symptoms effectively.
    Healthy lifestyle, physiotherapy, and support systems can improve the quality of life for patients.
    """
    response = qa_pipeline(question=question, context=context)
    return response['answer']
