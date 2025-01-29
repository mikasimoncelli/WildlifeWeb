@app.route('/admin/delete_sighting/<int:sighting_id>', methods=['POST'])
def delete_sighting(sighting_id):
    cursor = mysql.connection.cursor()
    print(sighting_id)

    sql = "SELECT SpeciesID FROM Sightings WHERE SightingID = %s"
    cursor.execute(sql, (sighting_id,))
    result = cursor.fetchone()

    if result:
        species_id = result[0] if isinstance(result, tuple) else result['SpeciesID']

        cursor.execute("DELETE FROM Sightings WHERE SightingID = %s", (sighting_id,))

        count_sql = "SELECT COUNT(*) FROM Sightings WHERE SpeciesID = %s"
        cursor.execute(count_sql, (species_id,))
        count_result = cursor.fetchone()
        count = count_result[0] if isinstance(count_result, tuple) else count_result['COUNT(*)']

        if count == 0:
            try:
                cursor.execute("DELETE FROM Species WHERE SpeciesID = %s", (species_id,))
            except mysql.connection.IntegrityError as e:
                print("Cannot delete species as it is still referenced by sightings.", e)
    
    mysql.connection.commit()

    return redirect(url_for('admin_sightings'))