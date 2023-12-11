from skillbridge import Workspace
import random

# Create pins at these locations on the design by default
PIN_LOCATIONS = [(8, 0),
                 (8, 15),]
                 # (),
                 # (),]

# expand rectangle a bit in each direction
RECT_SIZE = 20
EXTRA_LENGTH = 2
"""
Python class for creating arbitrary EM Structure in a layour view in cadence, and for
running simulations in EMX 
"""
class EMXSimulator():
    cv = None
    ws = None
    em_structure = []
    under_rect = None
    pins = []
    
    def __init__(self, server_id):
        self.ws = Workspace.open(server_id)
      
        # Currently this only works if you open the layout file first (pita, so trying to find a better way, debugging for over an hour hasn't led to
        # any tangible options, will continue later)
        self.cv = self.ws.ge.get_edit_cell_view()

        # Create underlying rectangle in M1_2B
        self.under_rect = self.ws.db.create_rect(self.cv, ["M1_2B", "drawing"], [(-100, -100), (400, 400)])
    
    # Creates a random 16x16 structure in the layout window
    def create_random_structure(self):
        # Grid of random 1's or zeros
        ex_grid = [[round(random.randint(0, 1)) for i in range(16)] for j in range(16)]
        
        
        
        # Create grid in LD layer
        for i in range(len(ex_grid)):
          for j in range(len(ex_grid[i])):
            if ex_grid[i][j] == 1 or (i, j) in PIN_LOCATIONS:
              # Create rectangle at this point in grid and add it to the group
              bottom_left = [(j*RECT_SIZE)-EXTRA_LENGTH, (i*RECT_SIZE)-EXTRA_LENGTH]
              top_right = [(j*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH, (i*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH]
              rect = self.ws.db.create_rect(self.cv, ["LD", "drawing"], [bottom_left, top_right])
        
              self.em_structure.append(rect)
        self.ws.le.hi_save()
        
    def create_pins(self):

        for (i, j) in PIN_LOCATIONS:
            
            bottom_left = [(j*RECT_SIZE)-EXTRA_LENGTH, (i*RECT_SIZE)-EXTRA_LENGTH]
            top_right = [(j*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH, (i*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH]
            # ws.le.create_pin(d_cellViewId 
            # l_layerPurposePair
            # t_shape 
            # l_points 
            # t_termName 
            # t_termDir 
            # l_accessDir)
            # ???? what is l_accessDir (double check this)
            in_pin_ld = self.ws.le.create_pin(self.cv, ["LD", "pin"], "rectangle", [bottom_left, top_right], "in1", "input", ["left", "right"])
            in_pin_m1 = self.ws.le.create_pin(self.cv, ["M1_2B", "pin"], "rectangle", [bottom_left, top_right], "in1_m", "input", ["left", "right"])
            
        
        # bottom_left = [(j*RECT_SIZE)-EXTRA_LENGTH, (i*RECT_SIZE)-EXTRA_LENGTH]
        # top_right = [(j*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH, (i*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH]
        # out_pin_ld = self.ws.le.create_pin(self.cv, ["LD", "pin"], "rectangle", [bottom_left, top_right], "out1", "input", ["left", "right"])
        # out_pin_m1 = self.ws.le.create_pin(self.cv, ["M1_2B", "pin"], "rectangle", [bottom_left, top_right], "out1_m", "input", ["left", "right"])

        self.ws.le.hi_save()

    # to be implemented
    def delete_structure(self):
        # deletion (doesn't seem to work for pins currently)
        while len(self.em_structure) > 0:
            # print(d_object.__dir__())
            self.ws.db.delete_object(self.em_structure.pop())

    # runs emx, provided that the EMX form is open in cadence
    def run_emx(self):
        self.ws["start_EMX"]()

    # extract s parameter values from output of EMX simulation
    def extract_sparams(self):
        self.ws["EMX_sparam_view"]()
        # for i in range(3):
        #     self.ws["CloseBox"]()



# Simulating in EMX
#show_EMX_gui(nil nil)
#hiiSetCurrentForm('EMXform3')
# EMXform3->Process->value="/home/zhengl/emxinterface/cadence6/emxinterface/processes/9HP_4222OLLD.proc"
#EMXform3->Start->value=30000000000.0
#EMXform3->Stop->value=100000000000.0
#EMXform3->Step->value=1000000000.0
#EMXform3->Signals->value="*"

#EMXform3->Cadence_pins->value= t
#EMXform3->Label_depth->value=1

# Run EMX simulation
# emx_gui = ws["show_EMX_gui"](None, None) # open emx gui

# y = ws.hi.getCurrentForm()
# print(dir(y))



        
