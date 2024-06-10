from skillbridge import Workspace
import random
import numpy as np


PIN_NAMES = ["LEFT", "RIGHT", "BOTTOM", "TOP"]

LAYER1 = "M1_2B" #M1_2B
LAYER2 = "LD" #LD

"""
Python class for creating arbitrary EM Structure in a layout view in cadence, and for
running simulations in EMX 
"""
class EMXSimulator():
    # Connects to the Cadence instance that is opened and run with skillbridge server_id
    def __init__(self, server_id, struct_height=300, struct_width=300):
        self.ws = Workspace.open(server_id)
      
        # Currently this only works if you open the layout file first
        self.cv = self.ws.ge.get_edit_cell_view()

        # Create underlying rectangle in M1_2B
        self.under_rect = ""
        self.pins_drawing = []
        self.pin_locations = []
        self.em_structure = []
        # 18x18 grid (16x16 plus ports)
        self.grid = [[0 for i in range(18)] for j in range(18)]

        #Sizing 
        # self.struct_size = struct_height
        self.struct_width = struct_width
        self.struct_height = struct_height
        
        # expand rectangle a bit in each direction
        self.rect_width = struct_width / 16
        self.extra_width = self.rect_width * 1/8
        self.rect_height = struct_height / 16
        self.extra_height = self.rect_height * 1/8
        # print(self.rect_width, self.rect_height)
        
    
    # Creates a random 16x16 structure in the layout window
    def create_random_structure(self):
        # Create underlying rectangle in M1_2B
        self.under_rect = self.ws.db.create_rect(self.cv, [LAYER1, "drawing"], [(-100, -100), (self.struct_width+125, self.struct_height+125)])
        self.em_structure.append(self.under_rect)
        
        # reinitialize grid of random 1's or zeros
        for i in range(1,17):
            for j in range(1, 17):
                self.grid[i][j] = random.randint(0, 1)
        
        # Create grid in LD layer
        for i in range(len(self.grid)):
          for j in range(len(self.grid[0])):
            if self.grid[i][j] == 1:
              # Create rectangle at this point in grid and add it to the group
              starting_height = self.rect_height * len(self.grid)
              bottom_left = [(j*self.rect_width)-self.extra_width, (-i*self.rect_height)+starting_height+self.extra_height]
              top_right = [(j*self.rect_width)+self.rect_width+self.extra_width, (-i*self.rect_height)+starting_height-self.rect_height-self.extra_height]
              rect = self.ws.db.create_rect(self.cv, [LAYER2, "drawing"], [bottom_left, top_right])
        
              self.em_structure.append(rect)

        self.ws["cliSave"](self.cv)

    # Creates a 18x18 structure with ports from the file name provided, and creates the pins
    def create_known_structure(self, structure, new_format=False):
        pin_locations = []
        data = np.array(structure)
        
        # extract pins
        # left,right,bottom,top
        left = (np.where(data[:, 0] == 1)[0][0], 0)
        right = (np.where(data[:, -1] == 1)[0][0], len(data)-1)
        bottom = (len(data)-1, np.where(data[-1, :] == 1)[0][0])
        top = (0, np.where(data[0, :] == 1)[0][0])
        pin_locations = [left, right, bottom, top]
    
        # reset sides to zeroes
        data[0, :] = 0
        data[len(data)-1, :] = 0
        data[:, 0] = 0
        data[:, len(data)-1] = 0
        self.grid = list(data)
        
        # Create underlying rectangle in M1_2B, add to structure list
        self.em_structure.append(self.ws.db.create_rect(self.cv, [LAYER1, "drawing"], [(-100, -100), (self.struct_width+125, self.struct_height+125)]))
        
        # Create grid in LD layer
        for i in range(len(self.grid)):
          for j in range(len(self.grid[0])):
            if self.grid[i][j] == 1:
              # Create rectangle at this point in grid and add it to the group
              starting_height = self.rect_height * len(self.grid)
              bottom_left = [(j*self.rect_width)-self.extra_width, (-i*self.rect_height)+starting_height+self.extra_height]
              top_right = [(j*self.rect_width)+self.rect_width+self.extra_width, (-i*self.rect_height)+starting_height-self.rect_height-self.extra_height]
              rect = self.ws.db.create_rect(self.cv, [LAYER2, "drawing"], [bottom_left, top_right])
        
              self.em_structure.append(rect)

        self.create_pins(pin_locations)

    # Create pins (ports) on the top, bottom, left, and right, one "pixel" outside the 16x16 grid
    # pin_locations should be passes as a list of (row, column) tuples if non-random locations
    # are desired
    def create_pins(self, pin_locations=None):
        # x = 0
        
        if pin_locations is None:
            #left
            self.pin_locations.append((random.randint(2, 15), 0))
            #right
            self.pin_locations.append((random.randint(2, 15), 17))
            #bottom
            self.pin_locations.append((17, random.randint(2, 15)))
            #top
            self.pin_locations.append((0, random.randint(2, 15)))
        else:
            #left, right, bottom, top, should be order of pins
            self.pin_locations = pin_locations


        
        for ((i, j), loc) in zip(self.pin_locations, PIN_NAMES):
            
            starting_height = self.rect_height * len(self.grid)
            bottom_left = [(j*self.rect_width)-self.extra_width, (-i*self.rect_height)+starting_height+self.extra_height]
            top_right = [(j*self.rect_width)+self.rect_width+self.extra_width, (-i*self.rect_height)+starting_height-self.rect_height-self.extra_height]


            rect_ld_1 = self.ws.db.create_rect(self.cv, [LAYER2, "drawing"], [bottom_left, top_right])
            rect_m1_1 = self.ws.db.create_rect(self.cv, [LAYER1, "drawing"], [bottom_left, top_right])
           
            # Draw rectangles in LD and M1_2B for pins
            rect_ld = self.ws.db.create_rect(self.cv, [LAYER2, "pin"], [bottom_left, top_right])
            rect_m1 = self.ws.db.create_rect(self.cv, [LAYER1, "pin"], [bottom_left, top_right])

            pin1_name = "{}".format(loc)
            net = self.ws.db.make_net(self.cv, pin1_name)
            pin1 = self.ws.db.create_pin(net, rect_ld)
            
            pin2_name = "{}_m".format(loc)
            net2 = self.ws.db.make_net(self.cv, pin2_name)
            pin2 = self.ws.db.create_pin(net2, rect_m1)
            self.pins_drawing.append(rect_ld_1)
            self.pins_drawing.append(rect_m1_1)
            self.pins_drawing.append(rect_ld)
            self.pins_drawing.append(rect_m1)
            # append rect_ld and rect_m1 too to delete them later?
            # x += 2

        for row, col in self.pin_locations:
            self.grid[row][col] = 1

        self.ws["cliSave"](self.cv)

    # deletes the 16x16 structure
    def delete_structure(self):
        # deletion (doesn't seem to work for pins currently)
        while len(self.em_structure) > 0:
            # print(d_object.__dir__())
            self.ws.db.delete_object(self.em_structure.pop())
            # note: doesn't delete self.grid
        # self.delete_rect()

    # Deletes pins (ports)
    def delete_pins(self):
        # delete all aspects of each pin
        # not sure if this is even necessary, but keeping it in case
        for term in self.cv.terminals:
            for pin in term.pins:
                self.ws.db.delete_object(pin)
                if pin.figs is not None:
                    for fig in pin.figs:
                        self.ws.db.delete_object(fig)
            self.ws.db.delete_object(term.net)

        # delete the pin drawing:
        for drawing in self.pins_drawing:
            self.ws.db.delete_object(drawing)

        for row, col in self.pin_locations:
            self.grid[row][col] = 0
        self.pin_locations = []
        self.pins_drawing = []
    
    # runs emx, provided that the EMX form is open in cadence
    def run_emx(self):
        self.ws["start_EMX"]()

    # Create a file to store the EM structure (i.e. ports and metal/empty locations)
    # Format is just writing 18x18=324 1's/0's to the file
    def create_struct_file(self, file):
        f = open(file, "w")
        # loop through row and columns of grid
        # write each row as a line
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):  
                f.write("{} ".format(self.grid[i][j]))
        f.close()

    # utility function for reading struct file and return 2D array
    def read_struct_file(self, file, rows, cols):
        f = open(file, "r")
        nums = f.read().split()
        to_return = [[0 for x in range(cols)] for y in range(rows)]
        for i in range(rows):
            for j in range(cols):
                to_return[i][j] = int(nums.pop(0))
                
        return to_return

    # utility function to close window in Cadence
    def close_window(self, window_num):
        window = self.ws["window"](window_num)
        self.ws.hi.close_window(window)

    # extract s parameter values from output of EMX simulation
    def extract_sparams(self):
        self.ws["EMX_sparam_view"]()
        # for i in range(3):
        #     self.ws["CloseBox"]()

        
