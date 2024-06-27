from xai_components.base import InArg, OutArg, Component, xai_component, secret, InCompArg
import googlemaps
from datetime import datetime
import json

@xai_component
class NearestTransitStop(Component):
    """A component that uses the Google Maps API to find the nearest public transit stop near a given location name.

    ##### inPorts:
    - api_key: Your Google API key
    - location_name: a String input representing the location name.

    ##### outPorts:
    - nearest_stop: a String output representing the nearest public transit stop.
    """
    api_key: InArg[secret]  
    location_name: InArg[str]
    closest_stations: OutArg[list]

    def execute(self, ctx) -> None:
        gmaps = googlemaps.Client(key=self.api_key.value)

        location = self.location_name.value

        geocode_result = gmaps.geocode(location)

        if geocode_result:
            lat = geocode_result[0]['geometry']['location']['lat']
            lng = geocode_result[0]['geometry']['location']['lng']

            nearest_transit = gmaps.places_nearby(location=(lat, lng), radius=1000, type='transit_station')

            if nearest_transit['results']:
                closest_stations = nearest_transit['results']
                self.closest_stations.value = [{'name': x['name'], 'location': {'lat': x['geometry']['location']['lat'], 'lng': x['geometry']['location']['lng']}} for x in closest_stations]
            else:
                self.closest_stations.value = []
        else:
            raise Exception("Location not found. Please provide a valid location name.")



@xai_component
class PublicTransportConnection(Component):
    """A component that uses the Google Maps Python API to find a connection between two public transport stations at a given time.

    ##### inPorts:
    - api_key: Your Google API key
    - origin: the origin public transport station.
    - destination: the destination public transport station.
    - departure_time: the time of departure in the format 'YYYY-MM-DD HH:MM:SS'.
    - arrival_time: the time of departure in the format 'YYYY-MM-DD HH:MM:SS'.


    ##### outPorts:
    - connection_info: information about the public transport connection.
    """
    api_key: InArg[secret]
    origin: InArg[str]
    destination: InArg[str]
    departure_time: InArg[str]
    arrival_time: InArg[str]

    connection_info: OutArg[str]

    def execute(self, ctx) -> None:
        gmaps = googlemaps.Client(key=self.api_key.value)

        origin = self.origin.value
        destination = self.destination.value
        departure_time = datetime.strptime(self.departure_time.value, '%Y-%m-%d %H:%M:%S') if self.departure_time.value is not None else None
        arrival_time = datetime.strptime(self.arrival_time.value, '%Y-%m-%d %H:%M:%S') if self.arrival_time.value is not None else None

        directions_result = gmaps.directions(origin, destination, mode="transit", departure_time=departure_time, arrival_time=arrival_time)

        if directions_result:
            result = directions_result[0]['legs'][0]
            for step in result['steps']:
                step.pop('polyline', None)
            self.connection_info.value = result
        else:
            self.connection_info.value = "No connection found."

@xai_component
class ExtractConnectionRequestFromJsonString(Component):
    json: InCompArg[str]
    origin: OutArg[str]
    destination: OutArg[str]
    departure_time: OutArg[str]
    arrival_time: OutArg[str]


    def execute(self, ctx) -> None:
        d = json.loads(self.json.value)
        print(d)
        self.origin.value = d['origin']
        self.destination.value = d['destination']
        self.departure_time.value = d.get('departure_time', None)
        self.arrival_time.value = d.get('arrival_time', None)