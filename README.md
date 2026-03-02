# Email-Automation
I tried to build a small email automation using SmolAgent(framework).
•	Created a fake Gmail account for testing.
•	The system  uses IMAP to fetch new emails from this account.(IMAP (Internet Message Access Protocol) as a high-tech "remote control" for your mailbox.)
•	If the email looks like a customer support email , then the system should automatically send a reply (like “Thank you, we’ll get back to you soon”).

Also added the .env file with all necessary credentials to access mail and sending replies.

# **THE IMPLEMENTATION STRICTLY COMPLIES WITH THE PRESCRIBED BOILERPLATE CODE TO ENSURE ARCHITECTURAL CONSISTENCY, RELIABILITY, AND HIGH-QUALITY PERFORMANCE.**

# Understand the tools , Tool Calling Agents and Manager agent .
In the SmolAgents framework,
- # *Tools* are the specific "atomic" actions an AI can perform. Think of these as the hands of the system.

- **fetch_support_emails**: This is our Data Retrieval tool. It uses IMAP to communicate with the Gmail server, filtering for specific keywords (help, issue, problem) and ensuring the system only processes new, unread messages.

- **generate_email_reply**: This is our Cognitive tool. It passes the email body to your local r1a1 model with a specific persona ("helpful support assistant") to craft a contextual response.

- **send_email_tool**: This is our Execution tool. It handles the complexities of SMTP, SSL contexts, and MIME formatting to actually deliver the response to the user.

# Tool Calling Agents -
Instead of having one giant AI try to do everything, we created Specialist Agents. This reduces "hallucinations" and increases reliability.

- **FetchAgent**: Dedicated entirely to monitoring the inbox. Its only "worldview" is the incoming mail stream.

- **ReplyAgent**: A creative specialist. It doesn't need to know how to log into Gmail; it only cares about taking a string of text and making it polite and helpful.

- **SendAgent**: A communications specialist. It ensures that once a reply is drafted, it is packaged correctly and sent to the right recipient.

# Manager Agent -
The CodeAgent (ManagerAgent) is the most critical part of your architecture. It doesn't perform the tasks itself; it manages them.

- **Orchestration**: It interprets the high-level goal ("Fetch 5 emails, reply, and send") and decides the sequence of events.

- **Reasoning** (Thought/Action/Observation): It uses a logic loop to check: "Did the FetchAgent find anything? Yes. Okay, now I need to give that data to the ReplyAgent."

- **Delegation**: It acts like a project manager, handing off specific sub-tasks to the specialist agents and collecting their outputs to move to the next step.

- **Error Handling & Planning**: With a max_steps=8 limit, it ensures the process doesn't run in an infinite loop and makes smart decisions if one specialist returns an error.


# WORKFLOW :
-tools - fetch email, generate reply, send email
agents - to use the tools.

main loop - check every 60 seconds for new emails, process first 5 unread emails.

THROUGH THIS EXAMPLE I UNDERSTOOD THE EACH TOOL MORE NICELY ALONG WITH RUNNING IT LOCALLY AND UNDERSTANDING SMOLAGENTS.
<img width="1564" height="680" alt="image" src="https://github.com/user-attachments/assets/36d3c2dd-11dd-40d0-b016-99addc666db7" />
