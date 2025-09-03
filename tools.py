import os
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
import asyncio
import re
import urllib.parse
from caching_utils import redis_cache

try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    FirecrawlApp = Any

class FlightSearchInput(BaseModel):
    origin: str = Field(description="Origin city or airport code (e.g., 'New York', 'JFK')")
    destination: str = Field(description="Destination city or airport code (e.g., 'Los Angeles', 'LAX')")
    departure_date: str = Field(description="Departure date in YYYY-MM-DD format")
    return_date: Optional[str] = Field(None, description="Return date in YYYY-MM-DD format for round trip")
    num_adults: int = Field(description="Number of adult passengers", default=1)
    formatted_output: Optional[bool] = Field(default=True, description="If True, return a formatted string for display; otherwise return JSON.")

class HotelSearchInput(BaseModel):
    destination: str = Field(description="Destination city for hotel search")
    check_in_date: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out_date: str = Field(description="Check-out date in YYYY-MM-DD format")
    num_adults: int = Field(description="Number of adult guests", default=1)
    formatted_output: Optional[bool] = Field(default=True, description="If True, return a formatted string for display; otherwise return JSON.")

def get_firecrawl_app() -> Optional[FirecrawlApp]:
    """Initialize Firecrawl app with API key"""
    if not FIRECRAWL_AVAILABLE:
        return None
    
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("WARNING: FIRECRAWL_API_KEY not found in environment variables")
        return None
    
    try:
        return FirecrawlApp(api_key=api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize FirecrawlApp: {e}")
        return None


@tool("firecrawl_flight_search", args_schema=FlightSearchInput)
@redis_cache(ttl=7200)
async def firecrawl_flight_search_tool(
    origin: str, 
    destination: str, 
    departure_date: str, 
    num_adults: int = 1,
    return_date: Optional[str] = None,
    formatted_output: Optional[bool] = True
) -> str:
    """
    Search for flights using Firecrawl web scraping with enhanced booking links.
    Returns comprehensive flight options with multiple booking site links.
    """
    app = get_firecrawl_app()
    if not app:
        if formatted_output:
            return " **Flight search unavailable** - Firecrawl not configured"
        return json.dumps({"error": "Firecrawl not available", "flights": []})
    
    try:
        def format_flight_markdown(flights: list, booking_links: dict) -> str:
            if not flights:
                return " **No flights found for your search**"
            
            lines = [
                f"## ✈ Flight Options: {origin} → {destination}",
                f" **Date:** {departure_date} | 👥 **Passengers:** {num_adults}",
                "",
                "###  Available Flights:"
            ]
            
            for i, flight in enumerate(flights[:8], 1):  # Limit to 8 flights
                price = flight.get('price', 'Price not available')
                airline = flight.get('airline', 'Unknown Airline')
                dep_time = flight.get('departure_time', 'N/A')
                arr_time = flight.get('arrival_time', 'N/A')
                duration = flight.get('duration', 'N/A')
                stops = flight.get('stops', 'N/A')
                
                lines.extend([
                    f"**{i}. {airline}**  {price}",
                    f"    {dep_time} → {arr_time} ({duration})",
                    f"    {stops}",
                    ""
                ])
            
            # Add booking links section
            lines.extend([
                "###  Book Your Flight:",
                ""
            ])
            
            for site, url in booking_links.items():
                lines.append(f" **[Search on {site}]({url})**")
            
            lines.extend([
                "",
                " *Tip: Compare prices across different sites for the best deals!*"
            ])
            
            return "\n".join(lines)
        
        # Generate booking links for multiple sites
        booking_links = generate_multiple_booking_links(origin, destination, departure_date, num_adults, return_date)
        
        # Use Skyscanner for scraping (most reliable)
        def format_for_skyscanner(loc):
            return loc.strip().replace(' ', '-').lower()
        
        origin_code = format_for_skyscanner(origin)
        dest_code = format_for_skyscanner(destination)
        
        # Format date for Skyscanner
        try:
            date_obj = datetime.strptime(departure_date, '%Y-%m-%d')
            skyscanner_date = date_obj.strftime('%y%m%d')
        except:
            skyscanner_date = departure_date.replace('-', '')[-6:]
        
        scrape_url = f"https://www.skyscanner.com/transport/flights/{origin_code}/{dest_code}/{skyscanner_date}/?adultsv2={num_adults}&cabinclass=economy"
        
        schema = {
            "type": "object",
            "properties": {
                "flights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "airline": {"type": "string"},
                            "flight_number": {"type": "string"},
                            "price": {"type": "string"},
                            "departure_time": {"type": "string"},
                            "arrival_time": {"type": "string"},
                            "departure_airport": {"type": "string"},
                            "arrival_airport": {"type": "string"},
                            "duration": {"type": "string"},
                            "stops": {"type": "string"},
                            "aircraft_type": {"type": "string"}
                        },
                        "required": ["airline", "price", "departure_time", "arrival_time"]
                    }
                }
            }
        }
        
        enhanced_prompt = f"""
        Extract comprehensive flight search results from this Skyscanner page for flights from {origin} to {destination} on {departure_date}.
        
        Find ALL available flight options including:
        - Different airlines (major carriers, budget airlines, etc.)
        - Various departure times throughout the day
        - Both direct and connecting flights
        - Different price points (economy, premium economy if shown)
        
        For each flight, extract:
        - Airline name (full name, not just code)
        - Flight number if visible
        - Total price with currency symbol
        - Departure time (format: HH:MM AM/PM)
        - Arrival time (format: HH:MM AM/PM)
        - Flight duration (e.g., "2h 30m")
        - Number of stops ("Direct" or "1 stop via XXX" or "2+ stops")
        - Aircraft type if available
        
        Prioritize actual bookable flights with real prices. Ignore ads, sponsored content, or incomplete listings.
        """
        
        print(f"🔍 Scraping flights from Skyscanner: {scrape_url}")
        
        result = app.scrape_url(
            url=scrape_url,
            formats=["extract"],
            extract={
                "schema": schema,
                "prompt": enhanced_prompt
            },
            timeout=90000,
            wait_for=8000
        )
        
        if hasattr(result, 'extract') and result.extract:
            data = result.extract
            if 'flights' in data and isinstance(data['flights'], list) and data['flights']:
                flights = data['flights'][:12]  # Limit to 12 flights
                
                # Add booking links to each flight
                for flight in flights:
                    flight['booking_links'] = booking_links
                
                if formatted_output:
                    return format_flight_markdown(flights, booking_links)
                
                return json.dumps({
                    "flights": flights,
                    "booking_links": booking_links,
                    "search_info": {
                        "origin": origin,
                        "destination": destination,
                        "departure_date": departure_date,
                        "return_date": return_date,
                        "num_adults": num_adults,
                        "total_found": len(flights),
                        "source": "Skyscanner"
                    }
                }, indent=2)
        
        # If no results, return with booking links
        if formatted_output:
            lines = [
                f"##  Flight Search: {origin} → {destination}",
                f" **Date:** {departure_date} | 👥 **Passengers:** {num_adults}",
                "",
                " **No specific flight data found, but you can search directly:**",
                ""
            ]
            for site, url in booking_links.items():
                lines.append(f"🎫 **[Search on {site}]({url})**")
            
            return "\n".join(lines)
        
        return json.dumps({
            "flights": [],
            "booking_links": booking_links,
            "message": "No flight data extracted, but booking links provided",
            "search_info": {
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "num_adults": num_adults
            }
        })
        
    except Exception as e:
        print(f" Flight search error: {e}")
        booking_links = generate_multiple_booking_links(origin, destination, departure_date, num_adults, return_date)
        
        if formatted_output:
            return f"""## ✈️ Flight Search Error
            
         **Search failed:** {str(e)}

        **But you can still search directly:**

        """ + "\n".join([f" **[Search on {site}]({url})**" for site, url in booking_links.items()])
        
        return json.dumps({
            "error": str(e),
            "flights": [],
            "booking_links": booking_links
        })

@tool("firecrawl_hotel_search", args_schema=HotelSearchInput)
@redis_cache(ttl=7200)
async def firecrawl_hotel_search_tool(
    destination: str,
    check_in_date: str,
    check_out_date: str,
    num_adults: int = 1,
    formatted_output: Optional[bool] = True
) -> str:
    """
    Search for hotels using Firecrawl web scraping with enhanced booking links.
    Returns comprehensive hotel options with multiple booking site links.
    """
    app = get_firecrawl_app()
    if not app:
        if formatted_output:
            return " **Hotel search unavailable** - Firecrawl not configured"
        return json.dumps({"error": "Firecrawl not available", "hotels": []})
    
    try:
        def format_hotel_markdown(hotels: list, booking_links: dict) -> str:
            if not hotels:
                return " **No hotels found for your search**"
            
            # Calculate nights
            try:
                checkin = datetime.strptime(check_in_date, '%Y-%m-%d')
                checkout = datetime.strptime(check_out_date, '%Y-%m-%d')
                nights = (checkout - checkin).days
            except:
                nights = 1
            
            lines = [
                f"##  Hotel Options in {destination}",
                f" **Check-in:** {check_in_date} | **Check-out:** {check_out_date} ({nights} nights)",
                f" **Guests:** {num_adults} adults",
                "",
                "###  Available Hotels:"
            ]
            
            for i, hotel in enumerate(hotels[:8], 1):  # Limit to 8 hotels
                name = hotel.get('name', 'Unknown Hotel')
                price = hotel.get('price_per_night', 'Price not available')
                rating = hotel.get('rating', 'N/A')
                location = hotel.get('location', '')
                amenities = hotel.get('amenities', '')
                reviews = hotel.get('total_reviews', '')
                
                lines.extend([
                    f"**{i}. {name}**  {price}",
                    f"    Rating: {rating}" + (f" ({reviews} reviews)" if reviews else ""),
                    f"    {location}" if location else "",
                    f"    {amenities}" if amenities else "",
                    ""
                ])
            
            # Add booking links section
            lines.extend([
                "###  Book Your Hotel:",
                ""
            ])
            
            for site, url in booking_links.items():
                lines.append(f" **[Search on {site}]({url})**")
            
            lines.extend([
                "",
                " *Tip: Check multiple sites for exclusive deals and compare amenities!*"
            ])
            
            return "\n".join(lines)
        
        # Generate booking links for multiple sites
        booking_links = generate_hotel_booking_links(destination, check_in_date, check_out_date, num_adults)
        
        # Use Booking.com for scraping
        destination_encoded = urllib.parse.quote_plus(destination)
        scrape_url = f"https://www.booking.com/searchresults.html?ss={destination_encoded}&checkin={check_in_date}&checkout={check_out_date}&group_adults={num_adults}&no_rooms=1"
        
        schema = {
            "type": "object",
            "properties": {
                "hotels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Hotel name"},
                            "price_per_night": {"type": "string", "description": "Price per night with currency"},
                            "rating": {"type": "string", "description": "Hotel rating (stars or score)"},
                            "location": {"type": "string", "description": "Hotel location/neighborhood"},
                            "amenities": {"type": "string", "description": "Key amenities"},
                            "review_score": {"type": "string", "description": "Guest review score"},
                            "total_reviews": {"type": "string", "description": "Number of reviews"},
                            "distance_from_center": {"type": "string", "description": "Distance from city center"},
                            "property_type": {"type": "string", "description": "Hotel, apartment, etc."}
                        },
                        "required": ["name", "price_per_night"]
                    }
                }
            }
        }
        
        enhanced_prompt = f"""
        Extract comprehensive hotel search results from this Booking.com page for {destination}.
        Dates: {check_in_date} to {check_out_date} for {num_adults} adults.
        
        Find diverse accommodation options:
        - Different price ranges (budget to luxury)
        - Various locations within the city
        - Different property types (hotels, apartments, B&Bs)
        - Different star ratings and guest scores
        
        For each property, extract:
        - Full hotel/property name
        - Price per night with currency (e.g., "$120/night", "₹8,500/night")
        - Star rating (1-5 stars) or guest review score
        - Specific location/neighborhood within {destination}
        - Key amenities (WiFi, pool, gym, breakfast, spa, parking, etc.)
        - Guest review score (e.g., "8.5/10", "Excellent")
        - Number of reviews if shown
        - Distance from city center
        - Property type (Hotel, Apartment, Resort, etc.)
        
        Focus on actual available properties with real pricing for the specified dates.
        Ignore ads, featured listings without prices, or incomplete entries.
        """
        
        print(f" Scraping hotels from Booking.com: {scrape_url}")
        
        result = app.scrape_url(
            url=scrape_url,
            formats=["extract"],
            extract={
                "schema": schema,
                "prompt": enhanced_prompt
            },
            timeout=90000,
            wait_for=8000
        )
        
        if hasattr(result, 'extract') and result.extract:
            data = result.extract
            if 'hotels' in data and isinstance(data['hotels'], list) and data['hotels']:
                hotels = data['hotels'][:10]  # Limit to 10 hotels
                
                # Add booking links to each hotel
                for hotel in hotels:
                    hotel['booking_links'] = booking_links
                
                if formatted_output:
                    return format_hotel_markdown(hotels, booking_links)
                
                return json.dumps({
                    "hotels": hotels,
                    "booking_links": booking_links,
                    "search_info": {
                        "destination": destination,
                        "check_in_date": check_in_date,
                        "check_out_date": check_out_date,
                        "num_adults": num_adults,
                        "total_found": len(hotels),
                        "source": "Booking.com"
                    }
                }, indent=2)
        
        # If no results, return with booking links
        if formatted_output:
            lines = [
                f"##  Hotel Search: {destination}",
                f" **Dates:** {check_in_date} to {check_out_date} | 👥 **Guests:** {num_adults}",
                "",
                " **No specific hotel data found, but you can search directly:**",
                ""
            ]
            for site, url in booking_links.items():
                lines.append(f"🏨 **[Search on {site}]({url})**")
            
            return "\n".join(lines)
        
        return json.dumps({
            "hotels": [],
            "booking_links": booking_links,
            "message": "No hotel data extracted, but booking links provided",
            "search_info": {
                "destination": destination,
                "check_in_date": check_in_date,
                "check_out_date": check_out_date,
                "num_adults": num_adults
            }
        })
        
    except Exception as e:
        print(f" Hotel search error: {e}")
        booking_links = generate_hotel_booking_links(destination, check_in_date, check_out_date, num_adults)
        
        if formatted_output:
            return f"""##  Hotel Search Error
            
 **Search failed:** {str(e)}

**But you can still search directly:**

""" + "\n".join([f" **[Search on {site}]({url})**" for site, url in booking_links.items()])
        
        return json.dumps({
            "error": str(e),
            "hotels": [],
            "booking_links": booking_links
        })

# Export the enhanced tools
ENHANCED_FIRECRAWL_TOOLS = [firecrawl_flight_search_tool, firecrawl_hotel_search_tool]

def are_firecrawl_tools_available() -> bool:
    """Check if Firecrawl tools can be used"""
    return FIRECRAWL_AVAILABLE and os.getenv("FIRECRAWL_API_KEY") is not None

# Test functions
async def test_flight_search():
    """Test the flight search functionality"""
    result = await firecrawl_flight_search_tool.ainvoke({
        "origin": "New York",
        "destination": "Los Angeles",
        "departure_date": "2025-02-15",
        "num_adults": 2,
        "formatted_output": True
    })
    print("Flight Search Test Result:")
    print(result)

async def test_hotel_search():
    """Test the hotel search functionality"""
    result = await firecrawl_hotel_search_tool.ainvoke({
        "destination": "Paris",
        "check_in_date": "2025-02-20",
        "check_out_date": "2025-02-24",
        "num_adults": 2,
        "formatted_output": True
    })
    print("Hotel Search Test Result:")
    print(result)

if __name__ == "__main__":
    print("Testing Enhanced Firecrawl Tools...")
    asyncio.run(test_flight_search())
    print("\n" + "="*50 + "\n")
    asyncio.run(test_hotel_search())
