//    Copyright (c) 2025
//    Author      : Bruno Capuano
//    Change Log  :
//
//    The MIT License (MIT)
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in
//    all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//    THE SOFTWARE.

using Microsoft.ML.OnnxRuntimeGenAI;
using System.Reflection.Emit;
using System.Reflection;

var userHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
var modelPath = Path.Combine(userHome, ".cache", @"modelscope\hub\LLM-Research\Phi-4-mini-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4");

var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";

// initialize model
var model = new Model(modelPath);
var tokenizer = new Tokenizer(model);

// chat start
Console.WriteLine(@"Ask your question. Type an empty string to Exit.");

// chat loop
while (true)
{
    // Get user question
    Console.WriteLine();
    Console.Write(@"Q: ");
    var userQ = Console.ReadLine();
    if (string.IsNullOrEmpty(userQ))
    {
        Console.WriteLine("Bye!");
        break;
    }

    // show phi response
    Console.Write("Phi4: ");
    var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{userQ}<|end|><|assistant|>";
    var tokens = tokenizer.Encode(fullPrompt);

    var generatorParams = new GeneratorParams(model);
    var sequences = tokenizer.Encode(fullPrompt);

    generatorParams.SetSearchOption("max_length", 2048);
    generatorParams.SetSearchOption("past_present_share_buffer", false);
    using var tokenizerStream = tokenizer.CreateStream();
    var generator = new Generator(model, generatorParams);

    generator.AppendTokens(tokens[0].ToArray());
    while (!generator.IsDone())
    {
        generator.GenerateNextToken();
        var output = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
        Console.Write(output);
    }
    Console.WriteLine();
}