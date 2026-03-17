# Roger AI Backend Architechture

We are using parallel fast agents for our small crucial work and one roger agent to manage all of that. It was a constant battle between intelligence and speed. 

### The Roger Agent, 

Roger agent is responsible for 
1. Generating the steps and plan
2. mantaining current state of task,
3. getting the latest context for spefic software. (grounding based on 4.) 

It can have two modes flash and pro. 

### Parallel Flash agents. 
We also have parallel flash agents, 
1. Box2D Agents - It is reponsible for giving box_2d array which tells where to draw the bounding box.
2. Feedback Agent - If the user has gone to another webpage (like youtube) it asks it come back to the task / or gives motivation if he is going right.
3. Tooltip Text Agent - This agent is reponsible for giving the text in the tooltip.
4. Identify software / website - Detect primary software needed to be utlised for the task user has asked for.

The parallel flash agents directly communicate with the UI and also update the context of roger agent (state of task). 

Let's take a look at flow with help of diagram, 
<img width="1440" height="1312" alt="image" src="https://github.com/user-attachments/assets/48c986a6-7fa8-4fea-b3c9-9d8dc5e052e3" />

User enters task  -> Roger agent (planning) (parallel p0)
                  -> Identify main softwares / website for task (parallel p0)

Step Wise Agent Triggers -> Box2D Agent  (Parallel p1)
                -> Tooltip Text Agent (Parallel p1) 
                -> Feedback Agents (Parallel p1)


Inputs are with increasing order of priority
1. Mouse Move (Screenshot Single + with mouse Position)
2. Click (Screenshot after 0.5/3 seconds depending on software or website)
3. Scroll (Screenshot after 0.5 seconds)
4. Drag / LongPress (Screenshot after 0.5 once the event has ended)
5. Type (Screenshot after 0.5 second after the event has ended)

This reduces the screenshots to very few numbers and are primarily used to community user activity to roger agent and feedback agents. 
<img width="1440" height="1102" alt="image" src="https://github.com/user-attachments/assets/359c4112-d726-4157-84ec-560ffa5dddf6" />


It is audio enabled and language agnostic 
https://github.com/TryRoger/backend/blob/23024d03e87e789b987d2cdc5976014511de3d14/controllers/copilot.py#L149

Task Agents 
https://github.com/TryRoger/backend/blob/main/controllers/v2/task_agents.py

FeedBack Agent 
https://github.com/TryRoger/backend/blob/main/controllers/v2/step_figure_agent.py

Roger Agent 
https://github.com/TryRoger/backend/blob/main/controllers/v2/roger_agent.py

Flow 
https://github.com/TryRoger/backend/blob/main/controllers/copilot.py

