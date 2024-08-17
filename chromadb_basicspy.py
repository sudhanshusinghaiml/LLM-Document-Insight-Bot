"""
For more tutorial information, you can refer to  https://docs.trychroma.com/guides
"""
import chromadb

def get_collections_from_chromadb():

    client = chromadb.Client()

    collections = client.create_collection(name= "my_collection")

    collections.add(
        documents= ["Finance is a term that addresses matters regarding the management, \
                    creation, and study of money and investments. It involves the use of credit and debt, \
                    securities, and investment to finance current projects using future income flows. \
                    Finance is closely linked to the time value of money, interest rates, and other related \
                    topics because of this temporal aspect",
                    "Public finance includes tax systems, government expenditures, budget procedures, \
                    stabilization policies and instruments, debt issues, and other government concerns. \
                    Corporate finance involves managing assets, liabilities, revenues, and debts for \
                    businesses. Personal finance defines all financial decisions and activities of an \
                    individual or household, including budgeting, insurance, mortgage planning, savings, \
                    and retirement planning",
                    "The BFSI sector is a critical element of the economy. Organizations of the BFSI sector \
                    help to enhance the possibility of the accumulation and circulation of capital, granting \
                    business owners to expand their businesses, and giving individuals the possibility to manage \
                    their finances properly.",
                    "Interestingly, the World Bank data shows that the percentage of global GDP, controlled by the \
                    BFSI sector, has been more than 20%. The BFSI sector carries out this important task by generating \
                    income, which helps to reduce the risks that can cause economic shocks in the country, and by \
                    ensuring that diversification of the economy is reached. In 2008, during one of the most significant \
                    financial crises of our times, the BFSI sector’s resilience helped prevent a complete economic collapse"
                    ],
        metadatas=[{"source":"investopedia"},{"source": "investopedia"}, {"source":"techcanvass"},{"source":"techcanvass"}],
        ids = ["id1","id2", "id3", "id4"]
    )

    return collections


if __name__ =="__main__":

    my_query = "What is public finance"

    collections = get_collections_from_chromadb()
    results = collections.query(
        query_texts= [my_query],
        n_results=1
    )
    
    print(results['documents'])