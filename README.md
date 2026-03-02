# Email-Automation
I tried to build a small email automation using SmolAgent(framework).
•	Created a fake Gmail account for testing.
•	The system  uses IMAP to fetch new emails from this account.(IMAP (Internet Message Access Protocol) as a high-tech "remote control" for your mailbox.)
•	If the email looks like a customer support email (for example: contains words like help, issue, problem), then the system should automatically send a reply (like “Thank you, we’ll get back to you soon”).

Also added the .env file with all necessary credentials to access mail and sending replies.

DETAILED OVERVIEW -----
Here we used our own LLM model r1a1
# WORKFLOW :
-tools - fetch email, generate reply, send email
agents - to use the tools.

main loop - check every 60 seconds for new emails, process first 5 unread emails.

THROUGH THIS EXAMPLE I UNDERSTOOD THE EACH TOOL MORE NICELY ALONG WITH RUNNING IT LOCALLY AND UNDERSTANDING SMOLAGENTS
<img width="1564" height="680" alt="image" src="https://github.com/user-attachments/assets/36d3c2dd-11dd-40d0-b016-99addc666db7" />
