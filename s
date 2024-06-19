{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "85a46e2d-8363-450c-8ebe-76b333d98055",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Preprocessed_Final_dataset.csv\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.preprocessing import LabelEncoder\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "9e84b6cf-38c1-4c65-b95c-48f5f51c0786",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>comment</th>\n",
       "      <th>label</th>\n",
       "      <th>clean_text</th>\n",
       "      <th>tokens</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>dalits are lowlives</td>\n",
       "      <td>N</td>\n",
       "      <td>dalits are lowlives</td>\n",
       "      <td>['dalits', 'lowlives']</td>\n",
       "      <td>dalits lowlives</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>gay people are a burden to our society</td>\n",
       "      <td>N</td>\n",
       "      <td>gay people are a burden to our society</td>\n",
       "      <td>['gay', 'people', 'burden', 'society']</td>\n",
       "      <td>gay people burden society</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Arabs are not welcome</td>\n",
       "      <td>N</td>\n",
       "      <td>arabs are not welcome</td>\n",
       "      <td>['arab', 'welcome']</td>\n",
       "      <td>arab welcome</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>I'm not saying we should actually eliminate he...</td>\n",
       "      <td>N</td>\n",
       "      <td>i am not saying we should actually eliminate h...</td>\n",
       "      <td>['say', 'actually', 'eliminate', 'heebs', 'wis...</td>\n",
       "      <td>say actually eliminate heebs wish naturally be...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>bananas are for black people</td>\n",
       "      <td>N</td>\n",
       "      <td>bananas are for black people</td>\n",
       "      <td>['bananas', 'black', 'people']</td>\n",
       "      <td>bananas black people</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                             comment label  \\\n",
       "0                                dalits are lowlives     N   \n",
       "1             gay people are a burden to our society     N   \n",
       "2                              Arabs are not welcome     N   \n",
       "3  I'm not saying we should actually eliminate he...     N   \n",
       "4                       bananas are for black people     N   \n",
       "\n",
       "                                          clean_text  \\\n",
       "0                                dalits are lowlives   \n",
       "1             gay people are a burden to our society   \n",
       "2                              arabs are not welcome   \n",
       "3  i am not saying we should actually eliminate h...   \n",
       "4                       bananas are for black people   \n",
       "\n",
       "                                              tokens  \\\n",
       "0                             ['dalits', 'lowlives']   \n",
       "1             ['gay', 'people', 'burden', 'society']   \n",
       "2                                ['arab', 'welcome']   \n",
       "3  ['say', 'actually', 'eliminate', 'heebs', 'wis...   \n",
       "4                     ['bananas', 'black', 'people']   \n",
       "\n",
       "                                                text  \n",
       "0                                    dalits lowlives  \n",
       "1                          gay people burden society  \n",
       "2                                       arab welcome  \n",
       "3  say actually eliminate heebs wish naturally be...  \n",
       "4                               bananas black people  "
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv('Preprocessed_Final_dataset.csv',encoding = 'latin1')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c6a9d7f-2a27-4d48-9adb-5fbb58a07f67",
   "metadata": {},
   "source": [
    "# TF-IDF Vectorizer"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c69c90a-9e13-4a28-84e7-ddbe53f489c1",
   "metadata": {},
   "source": [
    "### TF-IDF is the importance of a term is inversely related to its frequency across documents.TF gives us information on how often a term appears in a document and IDF gives us information about the relative rarity of a term in the collection of documents. By multiplying these values together we can get our final TF-IDF value.The higher the TF-IDF score the more important or relevant the term is; as a term gets less relevant, its TF-IDF score will approach 0."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "2251e14f-bc6c-441e-9a63-ab4aae618c5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Transform text to TF-IDF features with a limit on max features\n",
    "tfidf_vectorizer = TfidfVectorizer()\n",
    "X = tfidf_vectorizer.fit_transform(data['text'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8d85199-a3ad-4d53-9979-114d2bbee875",
   "metadata": {},
   "source": [
    "# Label Encoding:\n",
    "\n",
    "### LabelEncoder converts the categorical label into numeric labels.This is useful for classification tasks where the target variable needs to be in numerical format."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "4ad5854d-1d22-4167-a4bf-ec090775f498",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Encode the labels\n",
    "label_encoder = LabelEncoder()\n",
    "y = label_encoder.fit_transform(data['label'])\n",
    "tfidf_df['label'] = y\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "cab70d4c-63c4-4503-8891-89e15af309ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Concatenated DataFrame preview:\n",
      "                                             comment  label  \\\n",
      "0                                dalits are lowlives      N   \n",
      "1             gay people are a burden to our society      N   \n",
      "2                              Arabs are not welcome      N   \n",
      "3  I'm not saying we should actually eliminate he...      N   \n",
      "4                       bananas are for black people      N   \n",
      "\n",
      "                                          clean_text  \\\n",
      "0                                dalits are lowlives   \n",
      "1             gay people are a burden to our society   \n",
      "2                              arabs are not welcome   \n",
      "3  i am not saying we should actually eliminate h...   \n",
      "4                       bananas are for black people   \n",
      "\n",
      "                                              tokens  \\\n",
      "0                             ['dalits', 'lowlives']   \n",
      "1             ['gay', 'people', 'burden', 'society']   \n",
      "2                                ['arab', 'welcome']   \n",
      "3  ['say', 'actually', 'eliminate', 'heebs', 'wis...   \n",
      "4                     ['bananas', 'black', 'people']   \n",
      "\n",
      "                                                text  able  absolute  \\\n",
      "0                                    dalits lowlives   0.0       0.0   \n",
      "1                          gay people burden society   0.0       0.0   \n",
      "2                                       arab welcome   0.0       0.0   \n",
      "3  say actually eliminate heebs wish naturally be...   0.0       0.0   \n",
      "4                               bananas black people   0.0       0.0   \n",
      "\n",
      "   absolutely  abuse  accept  ...  would  wow  write  wrong  yeah  year  yes  \\\n",
      "0         0.0    0.0     0.0  ...    0.0  0.0    0.0    0.0   0.0   0.0  0.0   \n",
      "1         0.0    0.0     0.0  ...    0.0  0.0    0.0    0.0   0.0   0.0  0.0   \n",
      "2         0.0    0.0     0.0  ...    0.0  0.0    0.0    0.0   0.0   0.0  0.0   \n",
      "3         0.0    0.0     0.0  ...    0.0  0.0    0.0    0.0   0.0   0.0  0.0   \n",
      "4         0.0    0.0     0.0  ...    0.0  0.0    0.0    0.0   0.0   0.0  0.0   \n",
      "\n",
      "   yet  young  label  \n",
      "0  0.0    0.0      0  \n",
      "1  0.0    0.0      0  \n",
      "2  0.0    0.0      0  \n",
      "3  0.0    0.0      0  \n",
      "4  0.0    0.0      0  \n",
      "\n",
      "[5 rows x 1006 columns]\n"
     ]
    }
   ],
   "source": [
    "# Concatenate the original data (with clean text) and TF-IDF DataFrame\n",
    "result_df = pd.concat([data.reset_index(drop=True), tfidf_df], axis=1)\n",
    "print(\"Concatenated DataFrame preview:\")\n",
    "print(result_df.head())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
