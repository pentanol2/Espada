<h2>Spam web page detection</h2>
<p>Spam classifier is a script for classifying web page spamicity.
Spam web pages that try to attract traffic without being relevant to
the search query.</p>

<p>To check this we try to verify the conformity of the web page
heading with the content of the page body. We have a dataset of 
 urls under the name hostnames.txt. In this job the following
tasks are performed:</p>

<ol type="1">
    <li> <strong>Data Cleaning:</strong> We check the validity of the urls 
in our dataset by pinging each one of them using BeautifulSoup
library API and checking the http response code.</li>
<br>
<li><strong>Data Preprocessing:</strong> In this stage we crawl the web page content
corresponding the cleaned hostname list then the following two steps are performed:
<br><br>
<strong>Text Vectorization:</strong> Using tf-idf algorithm we perform text vectorization
for both the header and the body content of each page. The output is a couple of vectors corresponsing to 
word frequencies one for the heading the other for the content.
<br><br>
<strong>Cosine Similarity:</strong> To numerically quanity the score of each web page we apply the cosine similarity of the vector couples we got.
The corresponding new number we get is our preprocessed feature we will use in our classification problem.
</li>
<br>
<li><strong>Model Training:</strong> In this step we train a <strong>logistic regression</strong> model to classify 
our page to spam or not spam by mapping the Cosine Silimary to the actual labels column in our dataset.</li>
</ol>

