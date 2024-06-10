from skillbridge import Workspace
import random


PIN_NAMES = ["LEFT", "RIGHT", "BOTTOM", "TOP"]



LAYER1 = "M1_2B" #M1_2B
LAYER2 = "LD" #LD

"""
Python class for creating arbitrary EM Structure in a layour view in cadence, and for
running simulations in EMX 
"""
class EMXSimulator():
    
    def __init__(self, server_id, struct_size=300):
        self.ws = Workspace.open(server_id)
      
        # Currently this only works if you open the layout file first (pita, so trying to find a better way, debugging for over an hour hasn't led to
        # any tangible options, will continue later)
        self.cv = self.ws.ge.get_edit_cell_view()

        # Create underlying rectangle in M1_2B
        self.under_rect = self.ws.db.create_rect(self.cv, [LAYER1, "drawing"], [(-100, -100), (struct_size+125, struct_size+125)])
        self.pins_drawing = []
        self.pin_locations = []
        self.em_structure = []
        # 18x18 grid (16x16 plus ports)
        self.grid = [[0 for i in range(18)] for j in range(18)]

        #Sizing 
        self.struct_size = struct_size
        
        # expand rectangle a bit in each direction
        self.rect_size = struct_size / 16
        self.extra_length = self.rect_size * 1/8
        
    
    # Creates a random 16x16 structure in the layout window
    def create_random_structure(self):
        # reinitialize grid of random 1's or zeros
        for i in range(1,17):
            for j in range(1, 17):
                self.grid[i][j] = random.randint(0, 1)
            
        # for outer layer of pins
        # self.grid.insert(0, [0 for i in range(18)])
        # self.grid.append([0 for i in range(18)])
        
        # self.grid[8][0] = 1   # left
        # self.grid[8][17] = 1  # right
        # self.grid[17][8] = 1  # bottom
        # self.grid[0][8] = 1   # top
        
        # Create grid in LD layer
        for i in range(len(self.grid)):
          for j in range(len(self.grid[0])):
            if self.grid[i][j] == 1:
              # Create rectangle at this point in grid and add it to the group
              starting_height = self.rect_size * len(self.grid)
              bottom_left = [(j*self.rect_size)-self.extra_length, (-i*self.rect_size)+starting_height+self.extra_length]
              top_right = [(j*self.rect_size)+self.rect_size+self.extra_length, (-i*self.rect_size)+starting_height-self.rect_size-self.extra_length]
              rect = self.ws.db.create_rect(self.cv, [LAYER2, "drawing"], [bottom_left, top_right])
        
              self.em_structure.append(rect)

        self.ws["cliSave"](self.cv)

    # Create pins on the top, bottom, left, and right, one "pixel" outside the 16x16 grid
    # Currently hard-coded locations, but if pin deletion is figured out, should be random 1-16 location on side
    def create_pins(self):
        x = 0
        #left
        self.pin_locations.append((random.randint(2, 15), 0))
        #right
        self.pin_locations.append((random.randint(2, 15), 17))
        #bottom
        self.pin_locations.append((17, random.randint(2, 15)))
        #top
        self.pin_locations.append((0, random.randint(2, 15)))

        
        for ((i, j), loc) in zip(self.pin_locations, PIN_NAMES):
            
            starting_height = self.rect_size * len(self.grid)
            bottom_left = [(j*self.rect_size)-self.extra_length, (-i*self.rect_size)+starting_height+self.extra_length]
            top_right = [(j*self.rect_size)+self.rect_size+self.extra_length, (-i*self.rect_size)+starting_height-self.rect_size-self.extra_length]

            # in_pin_ld = self.ws.le.create_pin(self.cv, [LAYER2, "pin"], "rectangle", [bottom_left, top_right], loc, "input", ["top", "bottom", "left", "right"])
            # in_pin_m1 = self.ws.le.create_pin(self.cv, [LAYER1, "pin"], "rectangle", [bottom_left, top_right], loc + "_m", "input", ["top", "bottom", "left", "right"])

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
            x += 2

        for row, col in self.pin_locations:
            self.grid[row][col] = 1

        self.ws["cliSave"](self.cv)

    def delete_structure(self):
        # deletion (doesn't seem to work for pins currently)
        while len(self.em_structure) > 0:
            # print(d_object.__dir__())
            self.ws.db.delete_object(self.em_structure.pop())
            # note: doesn't delete self.grid

    def delete_rect(self):
        self.ws.db.delete_object(self.under_rect)

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
    # Format is just writing 18x18 grid of 1's/0's in the format:
    # row1 row2 row3 ...
    def create_struct_file(self, file):
        f = open(file, "w")
        # loop through row and columns of grid
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):  
                f.write("{} ".format(self.grid[i][j]))
        f.close()

    # utility function for read struct file and return 2D array (grid of metal)
    def read_struct_file(self, file, rows, cols):
        f = open(file, "r")
        nums = f.read().split()
        to_return = [[0 for x in range(cols)] for y in range(rows)]
        for i in range(rows):
            for j in range(cols):
                to_return[i][j] = int(nums.pop(0))
                
        return to_return

    # utility function to close window
    def close_window(self, window_num):
        window = self.ws["window"](window_num)
        self.ws.hi.close_window(window)

    # extract s parameter values from output of EMX simulation
    def extract_sparams(self):
        self.ws["EMX_sparam_view"]()
        # for i in range(3):
        #     self.ws["CloseBox"]()

        
