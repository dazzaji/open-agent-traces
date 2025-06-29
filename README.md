# OpenAI Agent SDK - Goal-to-Plan Open Source Implementation

## Overview
This project demonstrates a flexible, modular agent-based planning system built with the OpenAI Agent SDK. It provides a general approach to transform ANY initial goal or idea into a solid, well-structured plan with iterative refinement and evaluation.  I call this design approach "Agento" and you can see earlier versions, as well as other resources on the business, legal, and technical aspects of AI agents at [DazzaGreenwood.com](https://www.dazzagreenwood.com/).

OpenAI just launched a new open-source AI Agents SDK and I think it's the best agent framework out there. I was very fortunate to be granted early access and have been playing around with it, building imaginative ideas at nearly the speed of thought because its capabilities are powerful yet it's easy to work with.

I'd like to thank OpenAI for the early access to their Agents SDK. This project was initially built on a pre-release version and has been updated to work with the much-improved release version. The Dyson Sphere example shown in the code was suggested during a live demo at the AgentOps pre-release hack-night with OpenAI, which you can see here: https://x.com/AlexReibman/status/1899533549893746925

**OpenAI Announcement:** https://openai.com/index/new-tools-for-building-agents/  
**Documentation:** https://platform.openai.com/docs/guides/agents  
**SDK docs:** https://openai.github.io/openai-agents-python/  
**GitHub repo:** https://github.com/openai/openai-agents-python  
**Agent SDK video walkthrough and demo:** https://x.com/OpenAIDevs/status/1899531225468969240?t=617

## The General Approach

The true power of this system is its generality - it can handle virtually any goal or idea you provide, from business strategies to creative projects, from engineering designs to educational curricula. The system:

1. **Grounds the process with search**: Leverages knowledge to ensure plans are realistic and well-informed
2. **Establishes specific success metrics**: Tailored exactly to your unique goal
3. **Leverages multi-agent collaboration**: Different specialized agents handle different aspects of planning
4. **Provides iterative refinement**: Continuously improves plans based on evaluation against success criteria
5. **Offers complete modularity**: Each component can be swapped or modified independently

## Project Structure

The project consists of five modules that work together in a modular pipeline:

### Module 1: Criteria Generation (`module1.py`)
- Takes any user goal as input
- Uses a specialized agent to identify key success criteria specific to your goal
- Produces detailed reasoning and ratings for each criterion
- Creates a ranked list of criteria for project success

### Module 2: Plan Generation (`module2.py`)
- Takes the goal and success criteria from Module 1
- Uses a planning agent to generate multiple potential approaches
- Creates detailed outlines with reasoning for each approach
- Uses an evaluation agent to rank and select the best approach

### Module 3: Plan Expansion and Evaluation (`module3.py`)
- Takes the selected plan from Module 2
- Uses an expansion agent to flesh out each plan item in detail
- Evaluates each expanded item against the success criteria
- Creates a detailed evaluation summary for the expanded plan

### Module 4: Revision Identification (`module4.py`)
- Analyzes evaluation results from Module 3
- Identifies items that need revision based on criteria assessment
- Generates specific revision requests for items that don't fully meet criteria
- Evaluates potential impact of proposed revisions

### Module 5: Revision Implementation (`module5.py`)
- Takes approved revisions from Module 4
- Implements the revision requests into the plan items
- Evaluates how well the revisions address criteria
- Creates a final, revised plan with improvements

## Modularity and Interoperability

One of the most powerful features of this system is its modularity:

- **Framework Independence**: Each module can use a different agent framework (OpenAI Agents, AutoGen, Crew, LangGraph, etc.)
- **Standard Interfaces**: Modules communicate via standardized JSON formats
- **Plug-and-Play**: Replace any module with your own implementation as long as it respects the interface
- **Team Collaboration**: Different teams can work on different modules simultaneously
- **Experimentation**: Try different approaches for specific modules without disrupting the whole pipeline

This approach allows developers to leverage the strengths of various agent frameworks and work together efficiently even in distributed teams.

## Running the Project

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dazzaji/agento6.git
cd agento6
```

2. Create a virtual environment:
```bash
uv venv
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install openai-agents
uv pip install python-dotenv
```

4. Include your OpenAI API key in .env:

```
OPENAI_API_KEY=your_api_key_here
```

### Running the Modules

You can run each module sequentially:

```bash
python module1.py  # Input your goal or idea and get success criteria
python module2.py  # Creates and selects a plan
python module3.py  # Expands and evaluates the plan
python module4.py  # Identifies needed revisions
python module5.py  # Implements revisions into a final plan
```

Each module saves its output to a JSON file in the `data` directory, which serves as input for the next module.

There is extensive logging for each module as well, saved in a 'logs' directory.

Alternatively, you can create and run the entire pipeline:

```bash
python run_pipeline.py
```
## Making Your Life Easier with a Ready-to-Go Single File

To get you started fast, I’ve packed all of the OpenAI Agent SDK code and docs into one ready-to-use file. Just add or attach it to your LLM prompts for a seamless custom-agent-building experience. Grab the total Agent SDK in one file right [here](https://raw.githubusercontent.com/dazzaji/agento6/refs/heads/main/openai_openai-agents-python.md)!

## Possible Goals and Ideas

The best way to test Agento is to throw your own wonderful and weird ideas at it and see what it does, but to help get you started here are some interesting ideas you can use to try out the Agento system.  These are intended to be ideas or goals that an LLM agent system with the right tools and prompting and memory could make a plan to build then build, test, and deploy to the web as a working finished one-app business.  I made the ideas clear and definite enough so that they constitute a kind of high level specification with intended outcomes, but concise enough so they are stated as a single paragraph.  The sweet spot for this is being specific enough about the right things so that smart LLM agents could understand it, decompose it, extrapolate reasonably, turn it into a plan, and actually build and deploy it.  Naturally, to build and deploy some SWE modules and test/deploy modules (eg via GitHub and Vercel or Replit etc) would also be needed. The point of the first 5 modules is to take initial rough ideas to the point that they can server potentially as inputs to such build and deploy additional modules.


1. **Skill Swap Marketplace**  
   A web app where users list quirky skills they can teach (e.g., juggling, Morse code) and ones they want to learn, and the LLM matches them with others for live video sessions, generating custom lesson plans. It builds a simple profile-and-scheduling UI, launches as a commission-based platform, and targets lifelong learners seeking unique, human-to-human exchanges.

2. **Niche Podcast Finder**  
   An app that takes a user’s obscure interests (e.g., vintage typewriters, urban foraging) and scours the web and X for under-the-radar podcasts, delivering a curated list with summaries and direct links. The LLM designs a sleek, searchable interface with a “surprise me” button, deploys it as a free app with premium filters, and aims to hook audio enthusiasts tired of mainstream picks.

3. **Memory Mixtape Maker**  
   A web tool where users input a life event (e.g., first road trip, breakup), and the LLM crafts a personalized playlist with songs from that era or mood, paired with a short, evocative story about the memory. It integrates a music API, builds a shareable output page, and launches as a pay-per-play service for sentimental music lovers craving nostalgia.

4. **Remote Work Excuse Generator**  
   An app that takes a user’s job type and situation (e.g., missed deadline, bad Wi-Fi), then crafts witty, professional excuses or delay tactics, complete with email templates. The LLM pulls tone inspiration from X posts, creates a minimalist UI with a “randomize” feature, and deploys it as a freemium tool for gig workers and procrastinators needing a clever save.

5. **Micro-Adventure Planner**  
   A web app where users enter their location, free time, and vibe (e.g., chill, thrilling), and the LLM designs a bite-sized local adventure (e.g., a hidden trail, a pop-up event) with a map and checklist. It leverages web data and user reviews, builds a mobile-friendly UI, and launches as a subscription service for restless explorers stuck in a rut.

6. **Voice Note Storyteller**  
   A web app where users upload short voice memos about a day or feeling, and the LLM transforms them into polished, bite-sized short stories or poems with a selectable tone (e.g., whimsical, gritty). It builds a clean UI with audio playback and downloadable text, launches as a freemium tool with premium styles, and targets casual creatives who love a narrative twist.

7. **Hobby Budget Tracker**  
   An app that takes a user’s hobby (e.g., photography, knitting), income, and spending habits, then generates a tailored budget plan with tips to save or splurge smarter, pulling gear prices from the web. The LLM designs a dashboard with progress bars and alerts, deploys it as a free app with in-app purchases for custom reports, and aims at enthusiasts wanting financial control.

8. **Crowdsource My Outfit**  
   A web platform where users upload a photo of clothing items or describe an occasion, and the LLM curates outfit ideas by analyzing X posts and fashion blogs, offering a voting feature for public feedback. It builds a social, image-driven UI, launches as a subscription service with a free tier, and targets style-curious folks seeking validation or inspiration.

9. **Lost Item Sleuth**  
   An app where users describe a misplaced item (e.g., “blue scarf, last seen Tuesday”), and the LLM generates a step-by-step search plan based on psychology and common hiding spots, with a checklist and timer. It creates a minimalist, gamified interface, deploys as a one-time-purchase tool, and serves scatterbrained people desperate to declutter their chaos.

10. **Side Hustle Spark**  
    A web app that asks for a user’s skills, free time, and risk tolerance, then pitches three viable side hustle ideas with startup steps, market insights from web/X data, and a profitability calculator. The LLM builds an interactive proposal page with exportable plans, launches as a pay-per-use service, and targets ambitious go-getters hunting for extra cash.

## Contribution and Extension

I encourage you to experiment with this system:

- **Build your own modules**: Create alternative implementations of any stage
- **Make pull requests**: Share improvements or bug fixes
- **Report issues**: Let me know if you encounter problems
- **Share extensions**: If you build something cool based on this, shoot me a link!

This project is meant to demonstrate an approach to agent-based planning that can be extended and improved by the community.

## Roadmap

Some things I'm thinking about:

* Adding Claude 3.7 as a model
* Adding Perplexity for search
* Adding agents as tools
* Adding the traces to .log files or to their own files
* FastAPI with static site front-end maybe on Vercel or Replit
* Using endpoints to exchange JSON data between modules
* Streaming with `Runner.run_streamed()` for real-time updates as plans develop

## License

This project is licensed under Apache 2 License - see the LICENSE file for details.
