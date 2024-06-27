# MLCon2024: Xircuits Demo Agent

This project showcases how you can create a simple agent with Xircuits that has access to your Google Calendar and can find public transport routes for you.

## Prerequisites

1. **Python**: Ensure you have Python 3.11+ installed.
2. **Service Account**: Create a Google service account for accessing Google Calendar.
3. **Google Maps API Key**: Obtain a Google Maps API key for accessing public transit data.
3. **OpenAI Key**: Obtain an OpenAI key

## Setup Instructions

### 0. Install xircuits and dependencies

```
pip install uv
uv venv
source .venv/bin/activate
uv pip install xircuits
pip install -r requirements.txt
xircuits start
```

### 1. Create and Download Service Account Credentials

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing project.
3. Navigate to **APIs & Services > Credentials**.
4. Click on **Create credentials > Service account**.
5. Follow the prompts to create a service account.
6. Once created, click on the service account to edit it.
7. Go to the **Keys** tab and add a new key. Choose JSON format.
8. Download the `service_account.json` file and place it in the root directory of this project.

### 2. Share Calendar with Service Account

1. Open Google Calendar in your browser.
2. Select the calendar you want the agent to access.
3. Click on **Settings and sharing**.
4. Under **Share with specific people**, add the service account email (found in your `service_account.json` file).
5. Grant appropriate permissions (at least "Make changes to events").

### 3. Obtain Google Maps API Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Google Maps JavaScript API** and **Google Places API** for your project.
3. Navigate to **APIs & Services > Credentials**.
4. Click on **Create credentials > API key**.
5. Copy the generated API key into the Literal Secret Component that is connected to the public transit components.

### 4. Obtain OpenAI Key
1. Go to your OpenAI account and get an API Key
2. Copy the key into the Literal Secret Component that is connected to OpenAIAuthorize.


## Looking for a place to run this permanently?
At Xpress AI we are building a platform that allows anyone to create and host their agents.

Join us and many other builders on [xpress.ai](https://www.xpress.ai).
