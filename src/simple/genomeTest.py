#!/usr/local/bin/python2.7
# encoding: utf-8
'''
 -- Test genome description text files.

 This command takes as input the name of a genome description text file and then converts it into
 embeddings in a vector database.  The user can then enter queries on the console to ask questions.
 A blank line will be ignored, and a query of "/q" will end the program.


@author:     Bruce Parrello

@contact:    brucep.mobile@gmail.com
@deffield    updated: 2023-09-20
'''

import sys
import os
import re
import time
import openai
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from sqlalchemy.sql.expression import false
import textwrap

__all__ = []
__version__ = 0.1
__date__ = '2023-09-20'
__updated__ = '2023-09-20'

# In debug mode, exceptions are fatal and verbose defaults to ON
DEBUG = 1
PROFILE = 0
QUERY_MODEL = "gpt-4"

# These are global tuning parameters
verbose = false
tokens_max = 256
doc_max = 4
context = ""
sys_context = ""

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg


# Main method
def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by Bruce Parrello on %s.
  Copyright 2023 organization_name. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="if specified, more status messages will be printed")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        parser.add_argument('-d', '--docs', dest="docMax", action="store", default=4, type=int,
                            help="maximum number of documents per query to select from genome text file")
        parser.add_argument('-t', '--tokenMax', dest="tokMax", action="store", default=256, type=int,
                           help="maximum number of tokens to generate in the chat")
        parser.add_argument('-n', '--numVertex', dest="numVertex", action = "store", default=32, type=int,
                            help="number of neighbors to store in embedding database")
        parser.add_argument(dest="textFile", help="file containing the genome descriptive text", metavar="textFile")

        # Process arguments
        args = parser.parse_args()
        doc_max = args.docMax
        textFile = args.textFile
        verbose = args.verbose
        tokens_max = args.tokMax

        if (doc_max < 1) :
            print("Number of documents must be at least 1.")

        if verbose:
            print("Verbose mode on")

        ## Load the text file
        print(f"Input file for genome data is {textFile}.")
        loader = TextLoader(textFile);
        documents = loader.load();
        text_splitter = RecursiveCharacterTextSplitter(
            separators = ["\n"],
            keep_separator = False,
            chunk_size = 0,    # just splits on lines (separators)
            chunk_overlap  = 0,
            length_function = len,
            is_separator_regex = False,
        )
        texts = text_splitter.split_documents(documents)
        counter = len(texts)
        if verbose:
            print(f"{counter} lines of text found in data file.")
        ## Now "texts" is a list of all the text lines.  The first line describes a genome.  All the other lines
        ## describe features.  Next we build the embedding vectors from the text lines.  Those vectors are used
        ## to find relevant documents, and will be stored in "vectordb".
        if verbose:
            print("Creating embedding vectors from the document.")
        start = time.time()
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)
        if verbose:
            duration = time.time() - start
            print(f"{duration} seconds to load text with {counter} paragraphs.")
        # Create the relevant-document retriever.
        retriever = vectordb.as_retriever(search_kwargs={"k": doc_max})
        ## This will keep the responses for use as system context.
        sys_context = ""
        ## Now we loop through prompts from the user, submitting them as queries
        print("Type your first query, or ENTER for commands.")
        query = input()
        while (query != "/q") :
            # Only proceed if the query is nonblank.
            if (not query) :
                print("/q        -- terminate this session")
                print("/d NN     -- change document max to NN")
                print("/t NN     -- change token max to NN")
                print("/x        -- list the current values of the modifiable options")
                print("/clear    -- erase the current response list")
                print("/context  -- print the most recent context string")
            elif (query == "/x"):
                print(f"Token max is {tokens_max}.")
                print(f"Document max is {doc_max}.")
            elif (query == "/clear"):
                sys_size = len(sys_context) / 1024
                print(f"Erasing {sys_size:.2f}K of data stored in system context")
                sys_context = ""
            elif (query == "/context"):
                if (context == ""):
                    print("No context yet.")
                else:
                    print(textwrap.fill(context))
            elif (query.startswith("/")):
                m = re.match(r"/(.)\s+(.+)", query)
                command = m.group(1)
                parm = m.group(2)
                if (command == "d") :
                    # Here we need to change the number of documents retrieved.
                    try:
                        doc_max = int(parm)
                        print(f"Document search threshold is now {doc_max}.")
                        retriever = vectordb.as_retriever(search_kwargs={"k": doc_max})
                    except TypeError :
                        print(f"Invalid numeric value {parm}.")
                elif (command == "t") :
                    # Here we need to change the token limit.
                    try:
                        tokens_max = int(parm)
                        print(f"Chat token limit is now {tokens_max}.")
                    except TypeError :
                        print(f"Invalid numeric value {parm}.")
                else :
                    print(f"Invalid command code \"{command}\".")
            else :
                if verbose:
                    print(f"Processing query: {query}")
                # Get the relevant documents.
                docs = retriever.get_relevant_documents(query)
                # Form the documents into a context string.
                context = ""
                for doc in docs:
                    context += " " + doc.page_content
                if verbose:
                    found = len(docs)
                    print(f"{found} relevant documents found.")
                msgs = [ {"role": "system",    "content": sys_context},
                         {"role": "assistant", "content": context},
                         {"role": "user",      "content": query} ]
                response = openai.ChatCompletion.create(
                    model=QUERY_MODEL,
                    temperature=0.0,
                    max_tokens=tokens_max,
                    messages=msgs)
                response_string = response['choices'][0]['message']['content']
                print("--")
                print(response_string)
                print("--")
                # Update the system context
                sys_context += " " + response_string
            ## Ask for the next query.
            print("Type your next query, or ENTER for commands.")
            query = input()
        return 0
    except KeyboardInterrupt:
        print("Aborting execution after keyboard interrupt.")
        return 1
    except Exception as e:
        if DEBUG:
            raise(e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-v")
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = '_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())
