# !pip install llama-cpp-python

from llama_cpp import Llama
import json

user_prompt = """
Goal:
Generate 12–15 user stories for an online auction platform, categorised into functional and non-functional requirements.

Context:
The platform enables sellers to list products and create auctions of two types: bidding and buy-only.
- Auctions have a default seven-day duration, which sellers can shorten.
- For bidding auctions, sellers set a minimum bid value.
- Buyers can:
  - View the catalogue.
  - Place bids.
  - Delete bids.
  - View all bids for an auction.
- Buyers cannot edit bids or bid after the auction expires or a winner is determined.
- At auction end:
  - The system calculates the highest bid.
  - Marks it as the winner.
  - Notifies the seller via email.
- Auctions cannot be reactivated once concluded, but sellers can reuse previous auctions as templates.
- Authentication and authorisation are required for both buyers and sellers.
- Non-functional requirements:
  - System operates 24/7.
  - Auto-scales with load.
  - Encrypts data at rest and in transit.

Source:
Use the above context exclusively to create user stories.

Expectations:
- Provide 2 user stories, split into Functional and Non-Functional sections.
- Each story must follow the format:
  As a [role], I want [goal], so that [reason].
- Include clear, testable acceptance criteria for each story using Given/When/Then format.
- Organise output under two headings:
  Functional User Stories and Non-Functional User Stories.
- Ensure acceptance criteria are specific and measurable (e.g., “System uptime is 99.9%” rather than “System is reliable”).
"""

llm = Llama.from_pretrained(
        repo_id="MaziyarPanahi/gemma-3-12b-it-GGUF",
        filename="gemma-3-12b-it.Q6_K.gguf",
        #local_dir = "~/llm/models",
        #local_dir_use_symlinks=True,
        chat_format="llama-2",
        n_gpu_layers=99,
        verbose=True,
        n_ctx=4096,
)



response  = llm.create_chat_completion(
      messages = [
        {
            "role": "system",
            "content": "You are an Business SME who is championed in creating User Stories."
        },
        {
            "role": "user",
            "content": user_prompt,
          }
      ],
    temperature=0.7,
    max_tokens=-1,
    response_format={ "type": "json_object" }
)

#print(json.dumps([{k: response[k]} for k in response], indent=4))
#print(json.dumps(response, indent=4))
content = response["choices"][0]["message"]["content"]
print(content)




