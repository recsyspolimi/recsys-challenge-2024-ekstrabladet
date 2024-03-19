from neo4j import Driver

def delete_all(driver:Driver):
    with driver.session() as session:
        res = session.run("MATCH (n) DETACH DELETE n", database='neo4j').consume()
    return res.counters

def add_constraint(driver:Driver, label:str, key:str, debug=True):
    with driver.session() as session:
        res = session.run(f"CREATE CONSTRAINT {label}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE", database='neo4j').consume()
    if debug:
        print(res.counters)


def create_neo4j_admin_script(type:str):
    return ' '.join(
    [
        'bin/neo4j-admin database import full',
        #nodes
        f'--nodes=import/{type}/user_data.csv',
        f'--nodes=import/{type}/impression_data.csv',
        f'--nodes=import/{type}/article_data.csv',
        f'--nodes=import/{type}/category_data.csv',
        f'--nodes=import/{type}/entity_data.csv',
        f'--nodes=import/{type}/topic_data.csv',
        #relationships
        f'--relationships=import/{type}/user_to_impression.csv',
        f'--relationships=import/{type}/impression_to_article.csv',
        f'--relationships=import/{type}/impression_to_article_inview.csv',
        f'--relationships=import/{type}/impression_to_article_clicked.csv',
        f'--relationships=import/{type}/article_to_topic.csv',
        f'--relationships=import/{type}/article_to_entity.csv',
        f'--relationships=import/{type}/article_to_category.csv',
        
        '--overwrite-destination',
        f'newsdb-{type}'
        ])