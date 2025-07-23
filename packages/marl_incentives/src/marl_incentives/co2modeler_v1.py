# This code is to calculate the instantaneous carbon dioxide emission based on NGM dynamic model (Conlon and Lin, 2019)
# Conlon, J., & Lin, J. (2019). Greenhouse gas emission impact of autonomous
#  vehicle introduction in an urban network. Transportation Research Record,
#  2673(5), 142-152. doi: 10.1177/0361198119839970

# Input: velocity (m/s)
#        acceleration (m/s2)
#        type (light_passenger, light_van)
#        fuel (gasoline, diesel)

# Types of cars:
# light-duty passenger car: 1334kg (average weight of vehicles found in Helsinki)
# light-duty van: 1752kg

# Types of fuel:
# diesel: T_idle = 2660 gCO2/L; E_gas =  38e6 J/L;
# gasoline: T_idle = 2392 gCO2/L; E_gas =  31.5e6 J/L;

# Output: Instantaneous CO2 emission (g CO2)

# Other adjustable input:  Cd = Aerodynamic drag coefficient, [1]
#                          Crr = Rolling resistance, Conlon & Lin (2019)
#                          pho = Air density [kg/m3]
#                          T_idle = CO2 emission from gasoline [gCO2/gal], [2]
#                          E_gas = Energy in gasoline [kWh/gal]==>[3.6x10^6 kgm^2s-2/gal], [3]
#                          r = Regeneration efficiency ratio
#                          fuel_eff = Fuel efficiency (about 70%), [4]
#                           A = Frontal area [m2] (about 2.5 on average), [5]
#                           M = Vehicle mass [kg], [5]

# Footnote:
# [1] Wikipedia. url:https://en.wikipedia.org/wiki/Automobile_drag_coefficient#:~:text=The%20average%20modern%20automobile%20achieves,of%20body%20of%20the%20vehicle.
# [2] EPA. url: https://www.epa.gov/greenvehicles/greenhouse-gas-emissions-typical-passenger-vehicle#:~:text=Every%20gallon%20of%20gasoline%20burned%20creates%20about%208%2C887%20grams%20of%20CO2.
# [3] Bureau of Transportation Statistics. url: https://www.bts.gov/content/energy-consumption-mode-transportation-0
# [4] Based on real driving data
# [5] Based on Traficom database


def co2modeler(velocity: float, acceleration: float, type: str, fuel: str):
    if fuel == "gasoline":
        t_idle = 2392  # CO2 emission from gasoline [gCO2/L]
        e_gas = 31.5e6  # Energy in gasoline [J\L]
    elif fuel == "diesel":
        t_idle = 2660  # CO2 emission from diesel [gCO2/L]
        e_gas = 38e6  # Energy in diesel [J\L]

    if type == "light_passenger":
        m = 1334  # light-duty passenger vehicle mass [kg]
    elif type == "light_van":
        m = 1752  # light-duty van vehicle mass [kg]

    crr = 0.015  # Rolling resistance
    cd = 0.3  # Aerodynamic drag coefficient
    a = 2.5  # Frontal area [m2]
    g = 9.81  # Gravitational acceleration
    r = 0  # Regeneration efficiency ratio
    pho = 1.225  # Air density
    fuel_eff = 0.7  # fuel efficiency [70%]

    condition = (
        m * acceleration * velocity
        + m * g * crr * velocity
        + 0.5 * cd * a * pho * velocity**3
    )

    ei = t_idle / e_gas * condition

    if ei <= 0:
        E = r
    else:
        ei = ei * (velocity + 0.5 * acceleration)
        E = ei / fuel_eff

    return "%.2f" % E
