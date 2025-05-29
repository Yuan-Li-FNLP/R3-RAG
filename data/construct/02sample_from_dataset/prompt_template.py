def prompt_question_init(question):
   prompt_question_init = f'''
You are an expert in large language models and knowledge retrieval.

### Answer Format:
The problem analysis: [Provide a concise and logical analysis of the overall question, outlining what needs to be known or solved first to address the question effectively.]
The retrieval query: [Formulate a clear and specific query that the retrieval system should use to obtain the necessary information based on the problem analysis.]

### Instructions:
1. **Problem Analysis**:
   - Understand the main objective of the question.
   - Identify the key elements and any implicit information required.
   - Determine what needs to be known or solved first to approach the question.
   
2. **Retrieval Query**:
   - Based on the analysis, determine what specific information is needed next.
   - Formulate a precise question or set of keywords that the retrieval system can use to fetch relevant data.
   - Only a single query is permitted.

### Example:
#### Input:
When did the University of Oxford establish its first engineering department?
#### Output:
The problem analysis: The question seeks the founding year of the first engineering department at the University of Oxford. To answer this, we need to identify when the University of Oxford established its first engineering department.
The retrieval query: When was the first engineering department established at the University of Oxford?

### Your Task:
1. Your task is to analyze the following question and, using the provided format, output the first step analysis and a single retrieval query needed for that step. Remember, only provide the first step of the analysis without a complete solution.
2. Provide the entire response in English, regardless of the language of the input question and retrieved knowledge.
3. Your response shloud:
   - Only generate one step for the problem.
   - There is only one analysis and one query.
   - No iterative analysis - complete in a single step.
4. Your analysis should:
   - Be systematic, breaking the problem into manageable steps.
   - Be concise, focusing on relevant facts and logical inferences.
   - Be comprehensive, ensuring no critical detail is omitted.

#### Input:
{question}
#### Output:
'''
   return prompt_question_init

# 给出并列式和信息链的few shot的拆解教学
def prompt_question_initv2(question):
   prompt = f'''
# Role
You are an expert in large language models and knowledge retrieval, with extensive expertise in problem decomposition.

# Answer Format:
The problem analysis: [Analyze information based on the question.]
**Either:**
The retrieval query: [If information is insufficient, generate one query based on the analysis.]
**OR:**
The final answer: [If information is sufficient, provide the final answer based on the analysis.]

# Instructions:
1. **The Problem Analysis**:
   - **Decision Making**:
     - **Sufficient Information**: If the available information is adequate and no more query need to retrieve, use your parameter knowledge and reasoning capability to solve it and get the answer. Finaly, output the answer in the "The final answer:" section.
     - **Insufficient Information**: If the information is lacking, identify what information is missing:
       - **Determine Next Retrieval**: Consider what content should be retrieved next and decomposite the query into simpler retrieval queries.
2. **Response Formatting**:
   - **Start with Analysis**: Always begin with "The problem analysis:" followed by the detailed analysis. Only one analysis is permitted; therefore, include all your analytical content within this section.
   - **Choose Appropriate Section**: Based on the decision:
     - Include "The retrieval query:" if more information is needed.
     - Include "The final answer:" if sufficient information is available.
   - **Generate a Single Query**: Based on the analysis, formulate one targeted query and place it in the "The retrieval query:" section. Ensure that only one query is generated, focusing on the first retrieval question needed.
   - **Language Consistency**: Provide the entire response in English, regardless of the input language.

# Key Competencies
1. **Decision Making**:
   - **Assessment**: Evaluate whether it is feasible to resolve the entire problem based on all preceding steps.
   - **Action in Uncertainty**: If uncertain, initiate an attempt to solve the problem.
   - **Failure Handling**: If the solution process encounters a failure, terminate the attempt and analyze which information should be retrieved next.
2. **Decomposition of Retrieval Content**:
   - **Problem Breakdown**: When analysis requires solving a problem, decompose complex questions into smaller, actionable sub-questions (atomic problems).
   - **Step-by-Step Resolution**: Address these sub-questions sequentially and establish a logical sequence to answer the original question.
   - **Identifying Broad Problems**: If the analysis reveals that the information retrieved in the last step does not align with the query, this indicates that the problem is too broad. In such cases:
     - **Suggestion**: Propose a method to decompose the problem into smaller sub-questions.
     - **Implementation**: Attempt to break down the original query into simpler, more manageable questions.
   - **Methods of Decomposition**: Complex questions can be decomposed into **two fundamental forms**:
     - **Sequential Decomposition**:
       - **Definition**: Break the problem into a series of dependent steps, where each step relies on the result of the previous one.
     - **Parallel Decomposition**:
       - **Definition**: Split the problem into multiple independent sub-problems, solve each separately, and then combine the results.

# Few-shot learning
## Example1:  
### Input:
The question: Does the surface temperature range of the satellites of the earth overlap with the surface temperature range of Venus, and if so, what is the overlapping range?  

### Output:
The problem analysis: This is a **composite structure** because it requires parallel decomposition (finding the temperature ranges of the the satellites of the earth and Venus separately) followed by sequential decomposition (comparing the two ranges and calculating the overlap). First, perform the parallel decomposition by retrieving the surface temperature range of the the satellites of the earth.  
The retrieval query: What is the surface temperature range of the the satellites of the earth?  

## Example2:  
### Input:
The question: Who was the President of the United States when the founder of the best smartphone company in the US gave a speech at Stanford?

### Output:
The problem analysis: This is a composite structure because it requires parallel decomposition (identifying the founder of the best smartphone company in the US and finding out when they gave a speech at Stanford) followed by sequential decomposition (determining the President of the United States at that time). First, perform the parallel decomposition by retrieving information about the founder of the best smartphone company in the US.
The retrieval query: Who is the founder of the best smartphone company in the US?

## Example3:  
### Input:
The question: Who was the President of the United States when the founder of the best smartphone company in the US gave a speech at Stanford?

### Output:
The problem analysis: This is a composite structure because it requires parallel decomposition (identifying the founder of the best smartphone company in the US and finding out when they gave a speech at Stanford) followed by sequential decomposition (determining the President of the United States at that time). First, perform the parallel decomposition by retrieving information about the founder of the best smartphone company in the US.
The retrieval query: Who is the founder of the best smartphone company in the US?

#### Current Input:
The question: {question}

#### Output:
'''
   return prompt.strip()

# 给出并列式和信息链的few shot的拆解教学
def prompt_question(previous_steps, question):
    # Construct the Previous Steps section
    steps_formatted = ""
    for idx, step in enumerate(previous_steps, 1):
        steps_formatted += f"Step {idx}:\n"
        steps_formatted += f"The problem analysis: {step['analysis']}\n"
        steps_formatted += f"The retrieval query: {step['query']}\n"
        steps_formatted += f"The retrieval documents: {step['doc']}\n\n"

    prompt = f'''
You are an expert in large language models and knowledge retrieval.

### Answer Format:
The problem analysis: [Provide a concise and logical analysis of the current state of the problem, outlining what has been addressed and what remains to be solved.]
Either:
The retrieval query: [Formulate a clear and specific query that the retrieval system should use to obtain the necessary information based on the updated problem analysis.]
**OR**
The final answer: [Provide a clear and accurate answer to the original query based on the accumulated information.]

### Instructions:
1. **Problem Analysis**:
   - Review the previous analyses, retrieval queries, and retrieved documents.
   - Assess what has been resolved and identify any remaining gaps in information.
   - Determine if the accumulated information is sufficient to answer the question.
   - If information is insufficient:
       - Identify which aspect is lacking.
       - Consider if the problem is too broad and needs to be broken down into smaller, more specific sub-questions.
   - Provide a concise and logical analysis of the current state of the problem without repeating previous analyses.

2. **Decision Making**:
   - Based on the analysis, decide whether additional information is required.
   - If more information is needed:
       - Formulate a precise retrieval query for the next step.
       - If the problem is too broad, suggest a way to decompose it into smaller sub-questions.
   - If sufficient information is available:
       - Synthesize the information to provide the final answer.

3. **Response Formatting**:
   - Always start with "The problem analysis:" followed by the analysis.
   - Then, include either "The retrieval query:" or "The final answer:" based on the decision.
   - Ensure clarity and logical flow in the analysis and subsequent output.
   - Provide the response entirely in English, regardless of the input language.

### Example 1:
#### Input:
**The question**:
What are the primary factors contributing to the decline of honeybee populations?

**The Previous Steps**:
Step 1:
The problem analysis: To understand the decline in honeybee populations, we need to identify the major threats they face.
The retrieval query: What are the main threats to honeybee populations?
The retrieval documents: The primary threats to honeybee populations include pesticide exposure, habitat loss, diseases, parasites like Varroa mites, and climate change.

#### Output:
The problem analysis: The primary threats to honeybee populations have been identified, including pesticide exposure, habitat loss, diseases, parasites, and climate change. To determine which of these factors are most significant in contributing to the decline, we need more detailed information on their individual impacts.
The retrieval query: What is the impact of pesticide exposure on honeybee population decline compared to other factors?

### Example 2:
#### Input:
**The question**:
How does quantum entanglement influence the performance of quantum computers?

**The Previous Steps**:
Step 1:
The problem analysis: To understand the influence of quantum entanglement on quantum computers, we need to explore how entanglement affects computational processes.
The retrieval query: What role does quantum entanglement play in the functioning of quantum computers?
The retrieval documents: Quantum entanglement allows qubits to be interconnected in ways that classical bits cannot, enabling parallel processing and enhancing computational power for certain tasks.

#### Output:
The problem analysis: Quantum entanglement enables qubits to perform parallel processing and enhances computational power for specific tasks in quantum computers. With this information, we can conclude how entanglement directly influences the performance and efficiency of quantum computing systems.
The final answer: Quantum entanglement significantly enhances the performance of quantum computers by enabling qubits to perform parallel processing and solve complex problems more efficiently than classical computers. This interconnectedness allows for faster computation and the ability to handle tasks that are otherwise intractable for classical systems.

### Example 3:
#### Input:
**The question**:
Explain the economic impacts of artificial intelligence on the global job market.

**The Previous Steps**:
Step 1:
The problem analysis: To explain the economic impacts of artificial intelligence (AI) on the global job market, we need to identify the various ways AI influences employment.
The retrieval query: What are the economic impacts of artificial intelligence on employment?
The retrieval documents: AI can lead to job displacement in certain sectors, create new job opportunities in technology and AI development, and influence productivity and economic growth.

Step 2:
The problem analysis: While AI causes job displacement in some sectors and creates opportunities in others, we need more specific information on which sectors are most affected and how AI influences productivity metrics.
The retrieval query: Which sectors are most affected by AI-driven job displacement and what are the specific productivity changes caused by AI implementation?
The retrieval documents: Manufacturing and transportation sectors experience significant job displacement due to automation, while the technology sector sees growth in AI-related roles. AI implementation has increased productivity by automating repetitive tasks and enabling data-driven decision-making.

#### Output:
The problem analysis: AI-driven job displacement primarily affects the manufacturing and transportation sectors, while it creates growth opportunities in the technology sector. Additionally, AI implementation enhances productivity by automating repetitive tasks and enabling data-driven decision-making. To provide a comprehensive explanation of the economic impacts, we need to analyze the net effect on the global job market, considering both displacement and creation of jobs.
The retrieval query: What is the net effect of AI on global employment, considering both job displacement in manufacturing and transportation and job creation in the technology sector?

### Your Task:
1. Analyze the current state of the problem using the accumulated information from previous steps.
2. Decide whether to request additional information or provide the final answer.
3. Output the response following the specified format.
4. Ensure the response is concise, logically structured, and in English.

#### Current Input:
**The question**:
{question}

**The Previous Steps**:
{steps_formatted.strip()}

#### Output:
'''
    return prompt.strip()

# 给出并列式和信息链的few shot的拆解教学
def prompt_question_new(previous_steps, question):
   # Construct the Previous Steps section
   steps_formatted = ""
   for idx, step in enumerate(previous_steps, 1):
        steps_formatted += f"Step {idx}:\n"
        steps_formatted += f"The problem analysis: {step['analysis']}\n"
        steps_formatted += f"The retrieval query: {step['query']}\n"
        steps_formatted += f"The retrieval documents: {step['doc']}\n\n"

   prompt = f'''
# Role
You are an expert in large language models and knowledge retrieval, with extensive expertise in problem decomposition.

# Answer Format:
The problem analysis: [Analyze information based on previous steps and the question.]
**Either:**
The retrieval query: [If information is insufficient, generate one query based on the analysis.]
**OR:**
The final answer: [If information is sufficient, provide the final answer based on the analysis.]

# Instructions:
1. **The Problem Analysis**:
   - **Review Previous Steps**: Examine all prior steps, including analyses, queries, and documents.
   - **Decision Making**:
     - **Sufficient Information**: If the available information is adequate, analyze it to solve the problem and output the answer in the "The final answer:" section.
     - **Insufficient Information**: If the information is lacking:
       - **Evaluate Last Step**: Analyze the documents obtained in the last step to determine if they are relevant to the query or the problem at hand.
       - **Assess Previous Steps**: Identify what information is missing from the overall previous steps.
       - **Determine Next Retrieval**: Consider what content should be retrieved next and decomposite the query into simpler retrieval queries.
2. **Response Formatting**:
   - **Start with Analysis**: Always begin with "The problem analysis:" followed by the detailed analysis. Only one analysis is permitted; therefore, include all your analytical content within this section.
   - **Choose Appropriate Section**: Based on the decision:
     - Include "The retrieval query:" if more information is needed.
     - Include "The final answer:" if sufficient information is available.
   - **Generate a Single Query**: Based on the analysis, formulate one targeted query and place it in the "The retrieval query:" section. Ensure that only one query is generated, focusing on the first retrieval question needed.
   - **Language Consistency**: Provide the entire response in English, regardless of the input language.

# Key Competencies
1. **Decision Making**:
   - **Assessment**: Evaluate whether it is feasible to resolve the entire problem based on all preceding steps.
   - **Action in Uncertainty**: If uncertain, initiate an attempt to solve the problem.
   - **Failure Handling**: If the solution process encounters a failure, terminate the attempt and analyze which information should be retrieved next.
2. **Decomposition of Retrieval Content**:
   - **Problem Breakdown**: When analysis requires solving a problem, decompose complex questions into smaller, actionable sub-questions (atomic problems).
   - **Step-by-Step Resolution**: Address these sub-questions sequentially and establish a logical sequence to answer the original question.
   - **Identifying Broad Problems**: If the analysis reveals that the information retrieved in the last step does not align with the query, this indicates that the problem is too broad. In such cases:
     - **Suggestion**: Propose a method to decompose the problem into smaller sub-questions.
     - **Implementation**: Attempt to break down the original query into simpler, more manageable questions.
   - **Methods of Decomposition**: Complex questions can be decomposed into **two fundamental forms**:
     - **Sequential Decomposition**:
       - **Definition**: Break the problem into a series of dependent steps, where each step relies on the result of the previous one.
     - **Parallel Decomposition**:
       - **Definition**: Split the problem into multiple independent sub-problems, solve each separately, and then combine the results.

# Few-shot learning
## Example1:  
### Input:
The question: Does the surface temperature range of the satellites of the earth overlap with the surface temperature range of Venus, and if so, what is the overlapping range?  

The Previous Steps:
Step 1:
The problem analysis: This is a **composite structure** because it requires parallel decomposition (finding the temperature ranges of the the satellites of the earth and Venus separately) followed by sequential decomposition (comparing the two ranges and calculating the overlap). First, perform the parallel decomposition by retrieving the surface temperature range of the the satellites of the earth.  
The retrieval query: What is the surface temperature range of the the satellites of the earth?  
The retrieval documents: The Earth's satellites are located at varying distances from Earth, ranging from approximately 160 kilometers to over 35,786 kilometers

### Output:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the surface temperature range of the satellites of the Earth. We need to further decompose this problem by sequential decomposition. First, retrieve the names of the satellites of the Earth, and then retrieve the surface temperature ranges of these satellites.
The retrieval query: What are the satellites of the Earth?

## Example2:  
### Input:
The question: Does the surface temperature range of the satellites of the earth overlap with the surface temperature range of Venus, and if so, what is the overlapping range?  

The Previous Steps:
Step 1:
The problem analysis: This is a **composite structure** because it requires parallel decomposition (finding the temperature ranges of the the satellites of the earth and Venus separately) followed by sequential decomposition (comparing the two ranges and calculating the overlap). First, perform the parallel decomposition by retrieving the surface temperature range of the the satellites of the earth.  
The retrieval query: What is the surface temperature range of the the satellites of the earth?  
The retrieval documents: The Earth's satellites are located at varying distances from Earth, ranging from approximately 160 kilometers to over 35,786 kilometers

Step 2:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the surface temperature range of the satellites of the Earth. We need to further decompose this problem by sequential decomposition. First, retrieve the names of the satellites of the Earth, and then retrieve the surface temperature ranges of these satellites.
The retrieval query: What are the satellites of the Earth?
The retrieval documents: Earth has only one natural satellite, called the Moon.

### Output:
The problem analysis: The retrieval in Step 2 provides valid documents. Based on the sequential decomposition results of Step 2, the next step is to retrieve information about the surface temperature of the Moon.
The retrieval query: What is the surface temperature range of moon?  

## Example3:  
### Input:
The question: Does the surface temperature range of the satellites of the earth overlap with the surface temperature range of Venus, and if so, what is the overlapping range?  

The Previous Steps:
Step 1:
The problem analysis: This is a **composite structure** because it requires parallel decomposition (finding the temperature ranges of the the satellites of the earth and Venus separately) followed by sequential decomposition (comparing the two ranges and calculating the overlap). First, perform the parallel decomposition by retrieving the surface temperature range of the the satellites of the earth.  
The retrieval query: What is the surface temperature range of the the satellites of the earth?  
The retrieval documents: The Earth's satellites are located at varying distances from Earth, ranging from approximately 160 kilometers to over 35,786 kilometers

Step 2:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the surface temperature range of the satellites of the Earth. We need to further decompose this problem by sequential decomposition. First, retrieve the names of the satellites of the Earth, and then retrieve the surface temperature ranges of these satellites.
The retrieval query: What are the satellites of the Earth?
The retrieval documents: Earth has only one natural satellite, called the Moon.

Step 3:
The problem analysis: The retrieval in Step 2 provides valid documents. Based on the sequential decomposition results of Step 2, the next step is to retrieve information about the surface temperature of the Moon.
The retrieval query: What is the surface temperature range of moon?
The retrieval documents: The Moon’s surface temperature ranges from **-173°C to 127°C**.

### Output:
The problem analysis: The retrieval in Step 3 provides valid documents. Based on the parallel decomposition results of Step 1, the next step is to retrieve information about the surface temperature of the Venus.
The retrieval query: What is the surface temperature range of Venus?

## Example4:  
### Input:
The question: Does the surface temperature range of the satellites of the earth overlap with the surface temperature range of Venus, and if so, what is the overlapping range?  

The Previous Steps:
Step 1:
The problem analysis: This is a **composite structure** because it requires parallel decomposition (finding the temperature ranges of the the satellites of the earth and Venus separately) followed by sequential decomposition (comparing the two ranges and calculating the overlap). First, perform the parallel decomposition by retrieving the surface temperature range of the the satellites of the earth.  
The retrieval query: What is the surface temperature range of the the satellites of the earth?  
The retrieval documents: The Earth's satellites are located at varying distances from Earth, ranging from approximately 160 kilometers to over 35,786 kilometers

Step 2:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the surface temperature range of the satellites of the Earth. We need to further decompose this problem by sequential decomposition. First, retrieve the names of the satellites of the Earth, and then retrieve the surface temperature ranges of these satellites.
The retrieval query: What are the satellites of the Earth?
The retrieval documents: Earth has only one natural satellite, called the Moon.

Step 3:
The problem analysis: The retrieval in Step 2 provides valid documents. Based on the sequential decomposition results of Step 2, the next step is to retrieve information about the surface temperature of the Moon.
The retrieval query: What is the surface temperature range of moon?
The retrieval documents: The Moon’s surface temperature ranges from **-173°C to 127°C**.

Step 4:
The problem analysis: The retrieval in Step 3 provides valid documents. Based on the parallel decomposition results of Step 1, the next step is to retrieve information about the surface temperature of the Venus.
The retrieval query: What is the surface temperature range of Venus?
The retrieval documents: Venus’s surface temperature is **462°C** (approximately constant across the planet's surface).

### Output:
The problem analysis: The retrieval in Step 4 provides valid documents. Based on the sequential decomposition results of Step 1, the next step is to compare the surface temperature ranges of the Moon and Venus. The ranges do not overlap because the Moon’s maximum temperature (127°C) is lower than Venus’s constant temperature (462°C). Since there is no overlap between the temperature ranges of the Moon and Venus, the question of calculating the overlapping range does not apply.  
The final answer: No.

#### Current Input:
**The question**:
{question}

**The Previous Steps**:
{steps_formatted.strip()}

#### Output:
'''
   return prompt.strip()

# 给出并列式和信息链的few shot的拆解教学
def prompt_question_newv2(previous_steps, question):
   # Construct the Previous Steps section
   steps_formatted = ""
   for idx, step in enumerate(previous_steps, 1):
        steps_formatted += f"Step {idx}:\n"
        steps_formatted += f"The problem analysis: {step['analysis']}\n"
        steps_formatted += f"The retrieval query: {step['query']}\n"
        steps_formatted += f"The retrieval documents: {step['doc']}\n\n"

   prompt = f'''
# Role
You are an expert in large language models and knowledge retrieval, with extensive expertise in problem decomposition.

# Answer Format:
The problem analysis: [Analyze information based on previous steps and the question.]
**Either:**
The retrieval query: [If information is insufficient, generate one query based on the analysis.]
**OR:**
The final answer: [If information is sufficient, provide the final answer based on the analysis.]

# Instructions:
1. **The Problem Analysis**:
   - **Review Previous Steps**: Examine all prior steps, including analyses, queries, and documents.
   - **Decision Making**:
     - **Sufficient Information**: If the available information is adequate, analyze it to solve the problem and output the answer in the "The final answer:" section.
     - **Insufficient Information**: If the information is lacking:
       - **Evaluate Last Step**: Analyze the documents obtained in the last step to determine if they are relevant to the query or the problem at hand.
       - **Utilizing parameter knowledge**: If the retrieval results of the query are not relevant, check if the model's own parameter knowledge can response the query.
       - **Assess Previous Steps**: Identify what information is missing from the overall previous steps.
       - **Determine Next Retrieval**: Consider what content should be retrieved next and decomposite the query into simpler retrieval queries.
2. **Response Formatting**:
   - **Start with Analysis**: Always begin with "The problem analysis:" followed by the detailed analysis. Only one analysis is permitted; therefore, include all your analytical content within this section.
   - **Choose Appropriate Section**: Based on the decision:
     - Include "The retrieval query:" if more information is needed.
     - Include "The final answer:" if sufficient information is available.
   - **Generate a Single Query**: Based on the analysis, formulate one targeted query and place it in the "The retrieval query:" section. Ensure that only one query is generated, focusing on the first retrieval question needed.
   - **Language Consistency**: Provide the entire response in English, regardless of the input language.

# Key Competencies
1. **Decision Making**:
   - **Assessment**: Evaluate whether it is feasible to resolve the entire problem based on all preceding steps.
   - **Action in Uncertainty**: If uncertain, initiate an attempt to solve the problem.
   - **Failure Handling**: If the solution process encounters a failure, terminate the attempt and analyze which information should be retrieved next.
2. **Decomposition of Retrieval Content**:
   - **Problem Breakdown**: When analysis requires solving a problem, decompose complex questions into smaller, actionable sub-questions (atomic problems).
   - **Step-by-Step Resolution**: Address these sub-questions sequentially and establish a logical sequence to answer the original question.
   - **Identifying Broad Problems**: If the analysis reveals that the information retrieved in the last step does not align with the query, this indicates that the problem is too broad. In such cases:
     - **Suggestion**: Propose a method to decompose the problem into smaller sub-questions.
     - **Implementation**: Attempt to break down the original query into simpler, more manageable questions.
   - **Methods of Decomposition**: Complex questions can be decomposed into **two fundamental forms**:
     - **Sequential Decomposition**:
       - **Definition**: Break the problem into a series of dependent steps, where each step relies on the result of the previous one.
     - **Parallel Decomposition**:
       - **Definition**: Split the problem into multiple independent sub-problems, solve each separately, and then combine the results.

# Few-shot learning
## Example1:  
### Input:
The question: Does the surface temperature range of the satellites of the earth overlap with the surface temperature range of Venus, and if so, what is the overlapping range?  

The Previous Steps:
Step 1:
The problem analysis: This is a **composite structure** because it requires parallel decomposition (finding the temperature ranges of the the satellites of the earth and Venus separately) followed by sequential decomposition (comparing the two ranges and calculating the overlap). First, perform the parallel decomposition by retrieving the surface temperature range of the the satellites of the earth.  
The retrieval query: What is the surface temperature range of the the satellites of the earth?  
The retrieval documents: The Earth's satellites are located at varying distances from Earth, ranging from approximately 160 kilometers to over 35,786 kilometers

### Output:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the surface temperature range of the satellites of the Earth. We need to further decompose this problem by sequential decomposition. First, retrieve the names of the satellites of the Earth, and then retrieve the surface temperature ranges of these satellites.
The retrieval query: What are the satellites of the Earth?

## Example2:  
### Input:
The question: Who was the President of the United States when the founder of the best smartphone company in the US gave a speech at Stanford?

The Previous Steps:
Step 1:
The problem analysis: This is a composite structure because it requires parallel decomposition (identifying the founder of the best smartphone company in the US and finding out when they gave a speech at Stanford) followed by sequential decomposition (determining the President of the United States at that time). First, perform the parallel decomposition by retrieving information about the founder of the best smartphone company in the US.
The retrieval query: Who is the founder of the best smartphone company in the US?
The retrieval documents: None

### Output:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the founder of the best smartphone company in the US. However, utilizing the parameter knowledge, we know that the best smartphone company in the US is Apple. And the founder of Apple is Steve Jobs. Now, we need to find out when Steve Jobs gave a speech at Stanford.
The retrieval query: When did Steve Jobs give a speech at Stanford?

## Example3:  
### Input:
The question: Who was the President of the United States when the founder of the best smartphone company in the US gave a speech at Stanford?

The Previous Steps:
Step 1:
The problem analysis: This is a composite structure because it requires parallel decomposition (identifying the founder of the best smartphone company in the US and finding out when they gave a speech at Stanford) followed by sequential decomposition (determining the President of the United States at that time). First, perform the parallel decomposition by retrieving information about the founder of the best smartphone company in the US.
The retrieval query: Who is the founder of the best smartphone company in the US?
The retrieval documents: None

Step 2:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the founder of the best smartphone company in the US. However, utilizing the parameter knowledge, we know that the best smartphone company in the US is Apple. And the founder of Apple is Steve Jobs. Now, we need to find out when Steve Jobs gave a speech at Stanford.
The retrieval query: When did Steve Jobs give a speech at Stanford?
The retrieval documents: Steve Jobs delivered a famous commencement speech at Stanford University on June 12, 2005.

Step 3:
The problem analysis: The retrieval in Step 2 provides valid documents regarding the date of Steve Jobs' speech at Stanford. Now, based on the sequential decomposition, the next step is to identify who was the President of the United States in 2005.
The retrieval query: Who was the President of the United States in 2005?
The retrieval documents: The President of the United States in 2005 was George W. Bush.

### Output:
The problem analysis: The retrieval in Step 3 provides valid documents. Based on the results from Step 2, the next step is to confirm the President of the United States at the time of Steve Jobs' Stanford speech, which was in 2005. The President of the United States during that time was George W. Bush.
The final answer: George W. Bush.

#### Current Input:
The question: {question}

The Previous Steps:
{steps_formatted.strip()}

#### Output:
'''
   return prompt.strip()

# 给出并列式和信息链的few shot的拆解教学
def prompt_question_end(previous_steps, question):
   # Construct the Previous Steps section
   steps_formatted = ""
   for idx, step in enumerate(previous_steps, 1):
        steps_formatted += f"Step {idx}:\n"
        steps_formatted += f"The problem analysis: {step['analysis']}\n"
        steps_formatted += f"The retrieval query: {step['query']}\n"
        steps_formatted += f"The retrieval documents: {step['doc']}\n\n"

   prompt = f'''
# Role
You are an expert in large language models and knowledge retrieval, with extensive expertise in problem decomposition.

# Answer Format:
The problem analysis: [Analyze information based on previous steps and the question.]
**Either:**
The retrieval query: [If information is insufficient, generate one query based on the analysis.]
**OR:**
The final answer: [If information is sufficient, provide the final answer based on the analysis.]

# Instructions:
1. **The Problem Analysis**:
   - **Review Previous Steps**: Examine all prior steps, including analyses, queries, and documents.
   - **Decision Making**:
     - **Sufficient Information**: If the available information is adequate, analyze it to solve the problem and output the answer in the "The final answer:" section.
     - **Insufficient Information**: If the information is lacking:
       - **Evaluate Last Step**: Analyze the documents obtained in the last step to determine if they are relevant to the query or the problem at hand.
       - **Utilizing parameter knowledge**: If the retrieval results of the query are not relevant, check if the model's own parameter knowledge can response the query.
       - **Assess Previous Steps**: Identify what information is missing from the overall previous steps.
       - **Determine Next Retrieval**: Consider what content should be retrieved next and decomposite the query into simpler retrieval queries.
2. **Response Formatting**:
   - **Start with Analysis**: Always begin with "The problem analysis:" followed by the detailed analysis. Only one analysis is permitted; therefore, include all your analytical content within this section.
   - **Choose Appropriate Section**: Based on the decision:
     - Include "The retrieval query:" if more information is needed.
     - Include "The final answer:" if sufficient information is available.
   - **Generate a Single Query**: Based on the analysis, formulate one targeted query and place it in the "The retrieval query:" section. Ensure that only one query is generated, focusing on the first retrieval question needed.
   - **Language Consistency**: Provide the entire response in English, regardless of the input language.

# Key Competencies
1. **Decision Making**:
   - **Assessment**: Evaluate whether it is feasible to resolve the entire problem based on all preceding steps.
   - **Action in Uncertainty**: If uncertain, initiate an attempt to solve the problem.
   - **Failure Handling**: If the solution process encounters a failure, terminate the attempt and analyze which information should be retrieved next.
2. **Decomposition of Retrieval Content**:
   - **Problem Breakdown**: When analysis requires solving a problem, decompose complex questions into smaller, actionable sub-questions (atomic problems).
   - **Step-by-Step Resolution**: Address these sub-questions sequentially and establish a logical sequence to answer the original question.
   - **Identifying Broad Problems**: If the analysis reveals that the information retrieved in the last step does not align with the query, this indicates that the problem is too broad. In such cases:
     - **Suggestion**: Propose a method to decompose the problem into smaller sub-questions.
     - **Implementation**: Attempt to break down the original query into simpler, more manageable questions.
   - **Methods of Decomposition**: Complex questions can be decomposed into **two fundamental forms**:
     - **Sequential Decomposition**:
       - **Definition**: Break the problem into a series of dependent steps, where each step relies on the result of the previous one.
     - **Parallel Decomposition**:
       - **Definition**: Split the problem into multiple independent sub-problems, solve each separately, and then combine the results.

# Few-shot learning
## Example1:  
### Input:
The question: Does the surface temperature range of the satellites of the earth overlap with the surface temperature range of Venus, and if so, what is the overlapping range?  

The Previous Steps:
Step 1:
The problem analysis: This is a **composite structure** because it requires parallel decomposition (finding the temperature ranges of the the satellites of the earth and Venus separately) followed by sequential decomposition (comparing the two ranges and calculating the overlap). First, perform the parallel decomposition by retrieving the surface temperature range of the the satellites of the earth.  
The retrieval query: What is the surface temperature range of the the satellites of the earth?  
The retrieval documents: The Earth's satellites are located at varying distances from Earth, ranging from approximately 160 kilometers to over 35,786 kilometers

### Output:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the surface temperature range of the satellites of the Earth. We need to further decompose this problem by sequential decomposition. First, retrieve the names of the satellites of the Earth, and then retrieve the surface temperature ranges of these satellites.
The retrieval query: What are the satellites of the Earth?

## Example2:  
### Input:
The question: Who was the President of the United States when the founder of the best smartphone company in the US gave a speech at Stanford?

The Previous Steps:
Step 1:
The problem analysis: This is a composite structure because it requires parallel decomposition (identifying the founder of the best smartphone company in the US and finding out when they gave a speech at Stanford) followed by sequential decomposition (determining the President of the United States at that time). First, perform the parallel decomposition by retrieving information about the founder of the best smartphone company in the US.
The retrieval query: Who is the founder of the best smartphone company in the US?
The retrieval documents: None

### Output:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the founder of the best smartphone company in the US. However, utilizing the parameter knowledge, we know that the best smartphone company in the US is Apple. And the founder of Apple is Steve Jobs. Now, we need to find out when Steve Jobs gave a speech at Stanford.
The retrieval query: When did Steve Jobs give a speech at Stanford?

## Example3:  
### Input:
The question: Who was the President of the United States when the founder of the best smartphone company in the US gave a speech at Stanford?

The Previous Steps:
Step 1:
The problem analysis: This is a composite structure because it requires parallel decomposition (identifying the founder of the best smartphone company in the US and finding out when they gave a speech at Stanford) followed by sequential decomposition (determining the President of the United States at that time). First, perform the parallel decomposition by retrieving information about the founder of the best smartphone company in the US.
The retrieval query: Who is the founder of the best smartphone company in the US?
The retrieval documents: None

Step 2:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the founder of the best smartphone company in the US. However, utilizing the parameter knowledge, we know that the best smartphone company in the US is Apple. And the founder of Apple is Steve Jobs. Now, we need to find out when Steve Jobs gave a speech at Stanford.
The retrieval query: When did Steve Jobs give a speech at Stanford?
The retrieval documents: Steve Jobs delivered a famous commencement speech at Stanford University on June 12, 2005.

Step 3:
The problem analysis: The retrieval in Step 2 provides valid documents regarding the date of Steve Jobs' speech at Stanford. Now, based on the sequential decomposition, the next step is to identify who was the President of the United States in 2005.
The retrieval query: Who was the President of the United States in 2005?
The retrieval documents: The President of the United States in 2005 was George W. Bush.

### Output:
The problem analysis: The retrieval in Step 3 provides valid documents. Based on the results from Step 2, the next step is to confirm the President of the United States at the time of Steve Jobs' Stanford speech, which was in 2005. The President of the United States during that time was George W. Bush.
The final answer: George W. Bush.

#### Current Input:
The question: {question}

The Previous Steps:
{steps_formatted.strip()}

#### Output:
'''
   return prompt.strip()

# 给出并列式和信息链的few shot的拆解教学
def prompt_question_endv2(previous_steps, question):
   # Construct the Previous Steps section
   steps_formatted = ""
   for idx, step in enumerate(previous_steps, 1):
        steps_formatted += f"Step {idx}:\n"
        steps_formatted += f"The problem analysis: {step['analysis']}\n"
        steps_formatted += f"The retrieval query: {step['query']}\n"
        steps_formatted += f"The retrieval documents: {step['doc']}\n\n"

   prompt = f'''
# Role
You are an expert in large language models and knowledge retrieval, with extensive expertise in problem decomposition.

# Answer Format:
The problem analysis: [Step-by-step analysis based on previous steps and the question by decomposing the question.]
The final answer: [Provide the final answer based on the analysis.]

# Instructions:
1. **The Problem Analysis**:
   - **Review Previous Steps**: Examine all prior steps, including analyses, queries, and documents.
   - **Solve the question**:
     - Utilize reasoning abilities to decompose the problem into simpler sub-questions. Methods of decomposition include, but are not limited to, Sequential Decomposition and Parallel Decomposition.
     - Use both the retrieved information from previous steps and your internal knowledge.
     - Always attempt to provide an answer, even if uncertain.
2. **Response Formatting**:
   - **Start with Analysis**: Always begin with "**The problem analysis:**" followed by detailed analysis. Include all analytical content within this section.
   - **Provide Final Answer**: After the analysis, include "**The final answer:**" with the solution derived from the analysis.
   - **Language Consistency**: Provide the entire response in English, regardless of the input language.

# Key Competencies
1. **Decomposition of question**:
   - **Problem Breakdown**: When analysis requires solving a problem, decompose complex questions into smaller, actionable sub-questions (atomic problems).
   - **Step-by-Step Resolution**: Address these sub-questions sequentially and establish a logical sequence to answer the original question.
   - **Methods of Decomposition**: Complex questions can be decomposed into **two fundamental forms**:
     - **Sequential Decomposition**:
       - **Definition**: Break the problem into a series of dependent steps, where each step relies on the result of the previous one.
     - **Parallel Decomposition**:
       - **Definition**: Split the problem into multiple independent sub-problems, solve each separately, and then combine the results.
2. **Utilization of Internal Knowledge**:
   - When retrieval information is insufficient, leverage internal knowledge to infer or deduce answers.
   - Clearly indicate when internal knowledge is being used.

# Few-shot learning
## Example1:  
### Input:
The question: Does the surface temperature range of the satellites of the earth overlap with the surface temperature range of Venus, and if so, what is the overlapping range?  

The Previous Steps:
Step 1:
The problem analysis: This is a **composite structure** because it requires parallel decomposition (finding the temperature ranges of the the satellites of the earth and Venus separately) followed by sequential decomposition (comparing the two ranges and calculating the overlap). First, perform the parallel decomposition by retrieving the surface temperature range of the the satellites of the earth.  
The retrieval query: What is the surface temperature range of the the satellites of the earth?  
The retrieval documents: The Earth's satellites are located at varying distances from Earth, ranging from approximately 160 kilometers to over 35,786 kilometers

### Output:
The problem analysis: The retrieval in Step 1 does not provide the necessary information about the surface temperature range of the satellites of the Earth. We need to further decompose this problem by sequential decomposition. First, retrieve the names of the satellites of the Earth, and then retrieve the surface temperature ranges of these satellites. Using internal knowledge, Earth has only one natural satellite, called the Moon. Therefore, the Moon’s surface temperature range is the temperature range of Earth's satellite. The Moon’s surface temperature ranges from **-173°C to 127°C**. Based on the parallel decomposition results of Step 1, the next step is to obtain the surface temperature of Venus. Using internal knowledge, Venus’s surface temperature is **462°C** (approximately constant across the planet's surface). Based on the sequential decomposition results of Step 1, the next step is to compare the surface temperature ranges of the Moon and Venus. The ranges do not overlap because the Moon’s maximum temperature (127°C) is lower than Venus’s constant temperature (462°C). Since there is no overlap between the temperature ranges of the Moon and Venus, the question of calculating the overlapping range does not apply.
The final answer: No

### Example 2:
**Input:**
The question: Who was the President of the United States when the founder of the best smartphone company in the US gave a speech at Stanford?

The Previous Steps:
Step 1:
The problem analysis: This is a composite structure because it requires parallel decomposition (identifying the founder of the best smartphone company in the US and finding out when they gave a speech at Stanford) followed by sequential decomposition (determining the President of the United States at that time). First, perform the parallel decomposition by retrieving information about the founder of the best smartphone company in the US.
The retrieval query: Who is the founder of the best smartphone company in the US?
The retrieval documents: None

**Output:**
**The problem analysis:** The retrieval in Step 1 does not provide the necessary information about the founder of the best smartphone company in the US. However, utilizing internal knowledge, the best smartphone company in the US is Apple. The founder of Apple is Steve Jobs. Now, we need to find out when Steve Jobs gave a speech at Stanford. Using internal knowledge, Steve Jobs delivered a famous commencement speech at Stanford University on June 12, 2005. Based on the sequential decomposition, the next step is to identify who was the President of the United States in 2005. Using internal knowledge, the President of the United States in 2005 was George W. Bush.
**The final answer:** George W. Bush.

#### Current Input:
The question: {question}

The Previous Steps:
{steps_formatted.strip()}

#### Output:
'''
   return prompt.strip()

# 给出并列式和信息链的few shot的拆解教学
def prompt_question_endv3(previous_steps, question):
   # Construct the Previous Steps section
   steps_formatted = ""
   for idx, step in enumerate(previous_steps, 1):
        steps_formatted += f"Step {idx}:\n"
        steps_formatted += f"The problem analysis: {step['analysis']}\n"
        steps_formatted += f"The retrieval query: {step['query']}\n"
        steps_formatted += f"The retrieval documents: {step['doc']}\n\n"

   prompt = f'''
# Role
You are an expert in large language models and knowledge retrieval, with extensive expertise in problem decomposition.

# Answer Format:
The problem analysis: [Utilizing previous steps and parameter knowledge, step-by-step analysis the question by decomposing the question. Finally try all your effort to get the correct answer.]
The final answer: [Provide the final answer based on the analysis.]

# Instructions:
1. **Response Formatting**:
   - **Start with Analysis**: Always begin with "**The problem analysis:**" followed by detailed analysis. Include all analytical content within this section.
   - **Provide Final Answer**: After the analysis, include "**The final answer:**" with the solution derived from the analysis.
   - **Language Consistency**: Provide the entire response in English, regardless of the input language.
2. No retriever is available; only the model's internal knowledge may be used. Do not invoke a retriever, and therefore do not generate retrieval queries.

# Key Competencies
1. **Decomposition of question**:
   - **Problem Breakdown**: When analysis requires solving a problem, decompose complex questions into smaller, actionable sub-questions (atomic problems).
   - **Step-by-Step Resolution**: Address these sub-questions sequentially and establish a logical sequence to answer the original question.
   - **Methods of Decomposition**: Complex questions can be decomposed into **two fundamental forms**:
     - **Sequential Decomposition**:
       - **Definition**: Break the problem into a series of dependent steps, where each step relies on the result of the previous one.
     - **Parallel Decomposition**:
       - **Definition**: Split the problem into multiple independent sub-problems, solve each separately, and then combine the results.

#### Current Input:
The question: {question}

The Previous Steps:
{steps_formatted.strip()}

#### Output:
'''
   return prompt.strip()
