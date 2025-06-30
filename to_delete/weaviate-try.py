# import json
# import weaviate
# from weaviate.util import generate_uuid5

# # התחברות ל-Weaviate - גרסה ישנה תואמת ל-1.19.6
# client = weaviate.Client("http://localhost:8080")

# try:
#     print("מחובר ל-Weaviate:", client.is_ready())

#     # הגדרת הסכמה עבור המענים - בפורמט הישן
#     class_obj = {
#         "class": "Maane",  
#         "vectorizer": "text2vec-transformers",
#         "properties": [
#             {"name": "maane_name", "dataType": ["string"]},
#             {"name": "code_maane", "dataType": ["int"]},
#             {"name": "budgets", "dataType": ["object[]"], "nestedProperties": [
#                 {"name": "budget_code", "dataType": ["int"]},
#                 {"name": "budget_name", "dataType": ["string"]}
#             ]}
#         ]
#     }

#     # יצירת הסכמה (אם לא קיימת)
#     if not client.schema.exists("Maane"):
#         client.schema.create_class(class_obj)
#         print("סכמה נוצרה בהצלחה!")

#     # טעינת ה-JSON
#     with open('maane.json', 'r', encoding='utf-8') as file:
#         maane_list = json.load(file)

#     # העלאת המענים ל-Weaviate
#     print("מעלה מענים ל-Weaviate...")
    
#     # שימוש ב-batch לביצועים טובים יותר
#     client.batch.configure(batch_size=100)
#     with client.batch as batch:
#         for maane in maane_list:
#             batch.add_data_object(
#                 data_object=maane,
#                 class_name="Maane",
#                 uuid=generate_uuid5(str(maane['code_maane']))
#             )

#     print("כל המענים הועלו בהצלחה!")

#     # פונקציה לביצוע חיפוש
#     def search_maane(query, limit=5):
#         result = (
#             client.query
#             .get("Maane", ["maane_name", "code_maane", "budgets {budget_code budget_name}"])
#             .with_near_text({"concepts": [query]})
#             .with_additional(["certainty"])
#             .with_limit(limit)
#             .do()
#         )
#         return result['data']['Get']['Maane'] if 'data' in result and 'Get' in result['data'] else []

#     # שימוש בפונקציה המעודכנת
#     print("\n--- חיפוש מענים ---")
#     results = search_maane("זיכרון השואה")
    
#     if not results:
#         print("לא נמצאו תוצאות")
#     else:
#         for maane in results:
#             print(f"שם מענה: {maane['maane_name']}")
#             print(f"קוד מענה: {maane['code_maane']}")
#             print("תקציבים:")
#             if 'budgets' in maane and maane['budgets']:
#                 for budget in maane['budgets']:
#                     print(f"  - {budget['budget_name']} (קוד: {budget['budget_code']})")
#             else:
#                 print("  - אין תקציבים")
            
#             if '_additional' in maane and 'certainty' in maane['_additional']:
#                 print(f"ודאות: {maane['_additional']['certainty']:.2f}")
#             print("---")

#     # שימוש בתוצאות כהקשר ל-LLM
#     if results:
#         context = "\n".join([f"{maane['maane_name']}: קוד {maane['code_maane']}" for maane in results])
#         print("\nהקשר למודל שפה:")
#         print(context)

#     def check_vector(maane_id):
#         try:
#             result = client.data_object.get_by_id(maane_id, class_name="Maane", with_vector=True)
#             if 'vector' in result:
#                 print(f"וקטור נמצא עבור מענה {maane_id}. אורך הווקטור: {len(result['vector'])}")
#             else:
#                 print(f"לא נמצא וקטור עבור מענה {maane_id}")
#         except Exception as e:
#             print(f"שגיאה בבדיקת הווקטור: {e}")

#     def main():
#         # בדיקת תוצאה לדוגמה
#         query = "זיכרון השואה"
#         print(f"\n--- תוצאות חיפוש עבור: '{query}' ---")
#         results = search_maane(query)
        
#         if not results:
#             print("לא נמצאו תוצאות לחיפוש")
#             return
            
#         for maane in results:
#             print(f"שם מענה: {maane['maane_name']}")
#             print(f"קוד מענה: {maane['code_maane']}")
#             print("תקציבים:")
#             if 'budgets' in maane and maane['budgets']:
#                 for budget in maane['budgets']:
#                     print(f"  - {budget['budget_name']} (קוד: {budget['budget_code']})")
#             else:
#                 print("  - אין תקציבים")
                
#             if '_additional' in maane and 'certainty' in maane['_additional']:
#                 print(f"ודאות: {maane['_additional']['certainty']:.2f}")
#             print("---")

#         # בדיקת הווקטור של המענה הראשון
#         if results:
#             maane_id = generate_uuid5(str(results[0]['code_maane']))
#             check_vector(maane_id)

#     if __name__ == "__main__":
#         main()

# except Exception as e:
#     print(f"שגיאה: {e}")
# finally:
#     # אין צורך לסגור חיבור בגרסה הישנה
#     pass