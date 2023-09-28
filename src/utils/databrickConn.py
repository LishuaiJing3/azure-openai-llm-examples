class databricksConn:
    """get connection from databricks."""

    def __init__(
        self,
        db_host,
        db_token,
        db_warehouse_id,
        db_catalog="sample",
        db_schema="nyctaxi"
    ):
        """Initializes databricks connection."""

        from langchain.utilities import SQLDatabase

        # Create an databricks conn instance.
        self.db = SQLDatabase.from_databricks(
            catalog=db_catalog,
            schema=db_schema,
            host=db_host,
            api_token=db_token,
            warehouse_id=db_warehouse_id,
        )

    def get_db(self):
        """Returns the Databricks connection instance."""

        return self.db
