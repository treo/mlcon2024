from xai_components.base import InArg, OutArg, Component, xai_component, InCompArg
from google.oauth2 import service_account
import googleapiclient.discovery
from datetime import datetime
import json

@xai_component
class GoogleCalendarEvents(Component):
    """A component that retrieves events from Google Calendar for a specific day.

    ##### inPorts:
    - service_account_json: a path to your service account json file
    - calendar_id: id of the calendar you want to use (often just an email address)
    - day: a String port representing the day for which events are to be retrieved.

    ##### outPorts:
    - events: a List of Strings representing the events for the specified day.
    """
    service_account_json: InCompArg[str]
    calendar_id: InCompArg[str]
    day: InCompArg[str]
    events: OutArg[list]

    def execute(self, ctx) -> None:
        SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
        SERVICE_ACCOUNT_FILE = self.service_account_json.value
        CALENDAR_ID = self.calendar_id.value

        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)

        requested_day = self.day.value
        events_result = service.events().list(calendarId=CALENDAR_ID, timeMin=requested_day + 'T00:00:00Z', timeMax=requested_day + 'T23:59:59Z').execute()
        events = events_result.get('items', [])

        if not events:
            self.events.value = ["No events found for the specified day."]
        else:
            events_list = []
            for event in events:
                event_name = event['summary']
                start_time = event['start'].get('dateTime', event['start'].get('date'))
                end_time = event['end'].get('dateTime', event['end'].get('date'))
                location = event.get('location', '')
                participants = event.get('attendees', [])
                participants_list = [participant['email'] for participant in participants]
                event_str = f"{event_name} from {start_time} - {end_time}"
                if location:
                    event_str += f" ({location})"
                if participants_list:
                    event_str += f" with {', '.join(participants_list[:-1])}, and {participants_list[-1]}"
                events_list.append(event_str)
            self.events.value = events_list

@xai_component
class CreateCalendarEvent(Component):
    """A component that creates a new event in Google Calendar.

    ##### inPorts:
    - service_account_json: a path to your service account json file
    - calendar_id: id of the calendar you want to use (often just an email address)
    - summary: a String port representing the event name.
    - start_time: a String port representing the start time of the event in ISO format (e.g., '2022-01-01T10:00:00').
    - end_time: a String port representing the end time of the event in ISO format.
    - location: an optional String port representing the location of the event.
    - participants: an optional List of emails representing the participants of the event.

    ##### outPorts:
    - event_id: a String port representing the ID of the created event.
    """
    service_account_json: InCompArg[str]
    calendar_id: InCompArg[str]

    summary: InCompArg[str]
    start_time: InCompArg[str]
    end_time: InCompArg[str]
    location: InArg[str]
    participants: InArg[list]
    event_id: OutArg[str]

    def execute(self, ctx) -> None:
        SCOPES = ['https://www.googleapis.com/auth/calendar']
        SERVICE_ACCOUNT_FILE = self.service_account_json.value
        CALENDAR_ID = self.calendar_id.value

        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)

        event = {
            'summary': self.summary.value,
            'start': {'dateTime': self.start_time.value, 'timeZone': 'UTC'},
            'end': {'dateTime': self.end_time.value, 'timeZone': 'UTC'}
        }

        if self.location.value:
            event['location'] = self.location.value

        if self.participants.value:
            attendees = [{'email': participant} for participant in self.participants.value]
            event['attendees'] = attendees

        created_event = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        self.event_id.value = created_event['id']

@xai_component
class ExtractEventFromJsonString(Component):
    json: InCompArg[str]
    summary: OutArg[str]
    start_time: OutArg[str]
    end_time: OutArg[str]
    location: OutArg[str]
    participants: OutArg[list]

    def execute(self, ctx) -> None:
        d = json.loads(self.json.value)
        self.summary.value = d['summary']
        self.start_time.value = d['start_time']
        self.end_time.value = d['end_time']
        self.location.value = d.get('location')
        self.participants.value = d.get('participants')