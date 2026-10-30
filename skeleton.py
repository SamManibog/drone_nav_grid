from abc import ABC, abstractmethod
from typing import Callable
import asyncio
import multiprocessing

SIMULATION_PORT = 9000
OPERATOR_PORT = 9001

# data for an environment to be loaded by the simulation
# should contain:
#   - a heightmap for terrain
#   - a drone starting position
#   - a a human position
#   - a obstacles and their positions
class Environment:
    pass

class SimulationProcessHandle:
    def __init__(self, port: int, environment: Environment):
        print(f"todo: configure simulation")

    async def start(self):
        '''start the simulation'''
        print("todo: start the simulation")

    async def end(self):
        '''end the simulation'''
        print("todo: end the simulation")

# a lidar raycast
# start: the point at which the ray was initially cast
# end: the point at which the ray stopped
# found: if the ray collided with any object at the end point
class LidarRay:
    def __init__(self, start, end, found: bool):
        self.start = start
        self.end = end
        self.found = found

# the state of a drone, from the perspective of the operator
# this class is meant to be small and only contain basic details about the
# drone's current state
# NOT THE SAME AS DroneState
class OperatorDroneState:
    def __init__(self, position, next_waypoint, task):
        self.position = position
        self.next_waypoint = next_waypoint
        self.task = task

    def to_json(self):
        print("todo: convert operator drone state to json")

    @classmethod
    def from_json(cls, json_str: str):
        print("todo: construct drone operator state from json")

# class representing a command that can be sent to the drone from the operator
class IOperatorCommand(ABC):
    @abstractmethod
    def to_json(self):
        '''convert this command to json format'''
        pass

# operator-facing mediator for TCP communications between the operator and drone
class OperatorDroneBridgeClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None
        self.last_state = None
        self.last_fpv_frame = None

    async def start(self, retries: int = 20, delay: float = 0.5):
        '''Start connection with built-in readiness retry check'''
        for attempt in range(1, retries + 1):
            try:
                self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
                print(f"Successfully connected to drone server at {self.host}:{self.port}")
                return
            except (ConnectionRefusedError, OSError) as e:
                if attempt == retries:
                    raise ConnectionError(f"Failed to connect to drone server after {retries} attempts.") from e
                print(f"Connection refused ({e}). Retrying in {delay}s (Attempt {attempt}/{retries})...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)  # Exponential backoff capped at 5 seconds

    def get_last_state(self):
        return self.last_state

    def get_last_fpv_frame(self):
        return self.last_fpv_frame

    async def send_command(self, command: IOperatorCommand):
        if not self.writer:
            raise ConnectionError("Client is not connected to the drone server.")
        command_json = command.to_json() + "\n"
        self.writer.write(command_json.encode('utf-8'))
        await self.writer.drain()

    async def end(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()

# drone-facing mediator for TCP communications between the operator and drone
class OperatorDroneBridgeServer:
    def __init__(self, port: int):
        self.port = port
        self.server = None
        self.command_queue = asyncio.Queue()
        self.current_writer = None

    async def _handle_client(self, reader, writer):
        self.current_writer = writer
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                await self.command_queue.put(line.decode('utf-8').strip())
        except asyncio.CancelledError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    # call this function once to start the server
    # DO NOT AWAIT THIS FUNCTION
    async def start(self):
        self.server = await asyncio.start_server(self._handle_client, '127.0.0.1', self.port)
        async with self.server:
            await self.server.serve_forever()

    async def send_state(self, state: OperatorDroneState):
        if self.current_writer:
            state_data = state.to_json() + "\n"
            self.current_writer.write(state_data.encode('utf-8'))
            await self.current_writer.drain()

    async def send_fpv_frame(self, frame: bytes):
        if self.current_writer:
            self.current_writer.write(len(frame).to_bytes(4, 'big') + frame)
            await self.current_writer.drain()

    async def poll_commands(self):
        return await self.command_queue.get()

    async def end(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()

# class representing a command that can be sent to the drone simulation
class IDroneSimCommand(ABC):
    @abstractmethod
    def to_json(self):
        '''convert this command to json format'''
        pass

# mediator for TCP communications between the simulation and drone
class SimulationDroneBridgeClient:
    def __init__(self, port: int, on_recv: Callable):
        print(f"todo: configure simulation-drone bridge for port {port}")

    async def start(self):
        print("todo: start the simulation-drone bridge")

    def poll_lidar_rays(self):
        '''gets the next unhandled lidar ray or None if all rays have been handled'''
        print("todo: return unpolled lidar ray")

    def get_last_state(self):
        '''gets the last operator state of the drone'''
        print("todo: get last drone state")

    def get_last_fpv_frame(self):
        '''get the last camera frame sent by the drone, old frames shouldn't be stored'''
        print("todo: get fpv frame")

    async def send_command(self, command: IDroneSimCommand):
        command_json = command.to_json()
        print("todo: send command to drone")

    async def end(self):
        print("todo: close connections on the simulation-drone bridge")

class OperatorInterface:
    def __init__(self):
        print("todo: construct operator interface")

    # returns true if updates should continue, returns false otherwise
    def update(self, drone_bridge: OperatorDroneBridgeClient):
        '''draw the screen and handle updates based on the drone state and given fpv frame'''
        return True

# the state of the drone's logic
# this is the class that handles 
# NOT THE SAME AS OperatorDroneState
class DroneState:
    def __init__(self, sim_bridge: SimulationDroneBridgeClient, operator_bridge: OperatorDroneBridgeServer):
        self.sim_bridge = sim_bridge
        self.operator_bridge = operator_bridge

    def update(self):
        print("todo: update drone state")

# Drone's main function
def drone_main(sim_port: int, operator_port: int):
    async def main():
        drone_state = None

        def on_recv_sim_update():
            if drone_state:
                drone_state.update()

        sim_client = SimulationDroneBridgeClient(sim_port, on_recv_sim_update)
        operator_bridge = OperatorDroneBridgeServer(operator_port)
        try:
            drone_state = DroneState(sim_client, operator_bridge)

            # wait for simulation client to connect
            await sim_client.start()

            # start server
            asyncio.create_task(operator_bridge.start())

            await asyncio.Future()

        except KeyboardInterrupt:
            pass

    asyncio.run(main())

# App's main function
async def main():
    # todo: add more advanced/configurable way of creating an environment
    environment = Environment()

    # create and start the simulation in a separate process
    simulation = SimulationProcessHandle(SIMULATION_PORT, environment)
    await simulation.start()

    # spawn and run the drone process
    drone_process = multiprocessing.Process(target=drone_main, args=(SIMULATION_PORT, OPERATOR_PORT,), daemon=True)
    drone_process.start()

    # run the operator interface
    drone_bridge = OperatorDroneBridgeClient('127.0.0.1', OPERATOR_PORT)
    await drone_bridge.start()
    interface = OperatorInterface()

    try:
        while interface.update(drone_bridge):
            await asyncio.sleep(0.001) # Yield control if async
    finally:
        await drone_bridge.end()
        await simulation.end()
        drone_process.terminate()
        drone_process.join()
