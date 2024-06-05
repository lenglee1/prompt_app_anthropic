from flask import Flask, render_template, request, session, jsonify
import logging
import anthropic
import os

app = Flask(__name__)
# This is needed to use sessions
app.secret_key = os.urandom(24)
# Load the API key from environment variables
api_key = os.getenv('ANTHROPIC_API_KEY')

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to ensure roles alternate correctly
def ensure_role_alternation(messages):
    fixed_messages = []
    last_role = None
    for msg in messages:
        role = msg['role']
        if last_role is None or role != last_role:
            fixed_messages.append(msg)
            last_role = role
        else:
            # Skip this message since it has the same role as the previous one
            pass
    return fixed_messages

# Function to handle API calls
def anthropic_api_call(messages, system_instruction='Your task is to assist the user based on the provided instructions and context.'):
    try:
        logging.debug(f"Messages sent to Anthropic: {messages}")

        # Initialize the Anthropic client
        client = anthropic.Anthropic(api_key=api_key)

        # Ensure roles alternate correctly
        formatted_messages = ensure_role_alternation([
            {"role": msg['role'], "content": msg['content']}
            for msg in messages
        ])

        # Ensure the system instruction is a string
        if system_instruction is not None:
            system_instruction = str(system_instruction)
        
        # Log the system instruction
        logging.debug(f"System instruction (type: {type(system_instruction)}): {system_instruction}")

        # Send the messages to the API
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            system=system_instruction,
            messages=formatted_messages
        )

        # Log the raw API response
        logging.debug(f"Raw API response: {response}")

        # Extract content from the response
        content = ""
        if isinstance(response.content, list):
            content = "\n".join([item.text if hasattr(item, 'text') else str(item) for item in response.content]).strip()
        else:
            content = response.content.strip()

        logging.debug(f"API Response Content: {content}")
        return content
    except Exception as e:
        logging.error(f"API call failed: {e}")
        return None

# Function to generate the summary and the "act as" prompt
def generate_summary_and_prompt(messages):
    summary_instruction = (
        "Please summarize the key requirements provided by the user in bullet points. "
        "Based on these requirements, suggest an appropriate persona to 'act as' to fulfill the user's request. "
        "Act as a prompt generator for Anthropic. Use the suggested persona and summary to engineer a prompt that would yield the best and most desirable response from Anthropic. "
        "Each prompt should involve asking Anthropic to 'act as' the persona given. "
        "The prompt should be detailed and comprehensive and should build on what was requested to generate the best possible response from Anthropic. "
        "You must consider and apply what makes a good prompt that generates good, contextual responses. "
        "You must give a summary of the key requirements, a suggested persona, and output the prompt you want to use."
    )
    messages.append({"role": "user", "content": summary_instruction})
    summary_response = anthropic_api_call(messages, system_instruction=summary_instruction)
    return summary_response

# Function to generate the final response based on the suggested prompt
def generate_final_response(messages, suggested_prompt):
    final_instruction = (
        f"Based on the following summary and suggested persona, provide the final response:\n\n"
        f"{suggested_prompt}"
    )
    messages.append({"role": "user", "content": final_instruction})
    final_response = anthropic_api_call(messages, system_instruction=final_instruction)
    return final_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_message = data.get('prompt')
        logging.info(f"Received prompt: {user_message}")

        if not user_message:
            return jsonify({'error': 'No prompt provided'}), 400

        if 'messages' not in session:
            session['messages'] = []

        session['messages'].append({"role": "user", "content": user_message})

        if len(session['messages']) > 1 and session['messages'][-2]['role'] == 'assistant':
            # The last AI response should have been a follow-up question, so use the user's response to generate the next message
            response = anthropic_api_call(session['messages'], system_instruction="You are an AI assistant.")

            if response:
                session['messages'].append({"role": "assistant", "content": response})
                session.modified = True

                # Generate summary and "act as" prompt
                summary_response = generate_summary_and_prompt(session['messages'])
                logging.info(f"Summary Response: {summary_response}")

                if summary_response:
                    session['messages'].append({"role": "assistant", "content": summary_response})
                    session.modified = True
                    # Generate final response based on the summary and suggested prompt
                    final_response_content = generate_final_response(session['messages'], summary_response)
                    logging.info(f"Final response: {final_response_content}")
                    if final_response_content:
                        return jsonify({'summary': summary_response, 'final_response': final_response_content})
                    else:
                        logging.error('Final response generation failed')
                        return jsonify({'error': 'Final response generation failed'}), 500
                else:
                    logging.error('Summary response was empty')
                    return jsonify({'error': 'Summary generation failed'}), 500
            else:
                logging.error('API call failed')
                return jsonify({'error': 'API call failed'}), 500
        else:
            # Generate a prompt to ask for more details if needed
            instructions = (
                "Help the user develop a clear set of requirements. "
                "Use language of a 3rd grade reading level"
                "Ask exactly three questions to gather the necessary information to fulfill the user's request."
            )

            # Send the conversation history to Anthropic and get a response
            response = anthropic_api_call(session['messages'], system_instruction=instructions)

            # Add Anthropic's response to the conversation history
            if response:
                session['messages'].append({"role": "assistant", "content": response})
                session.modified = True
                logging.info(f"API response: {response}")
                return jsonify({'response': response})
            else:
                logging.error('API call failed')
                return jsonify({'error': 'API call failed'}), 500
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)