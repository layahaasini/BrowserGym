-- Dumping data for table `oc_t_item_location`
LOCK TABLES `oc_t_item_location` WRITE;
/*!40000 ALTER TABLE `oc_t_item_location` DISABLE KEYS */;
DELETE FROM `oc_t_item_location` WHERE fk_i_item_id >= 84143;
/*!40000 ALTER TABLE `oc_t_item_location` ENABLE KEYS */;
UNLOCK TABLES;
LOCK TABLES `oc_t_item_resource` WRITE;
/*!40000 ALTER TABLE `oc_t_item_resource` DISABLE KEYS */;
DELETE FROM `oc_t_item_resource` WHERE fk_i_item_id >= 84143;
/*!40000 ALTER TABLE `oc_t_item_resource` ENABLE KEYS */;
UNLOCK TABLES;
LOCK TABLES `oc_t_item_stats` WRITE;
/*!40000 ALTER TABLE `oc_t_item_stats` DISABLE KEYS */;
DELETE FROM `oc_t_item_stats` WHERE fk_i_item_id >= 84143;
/*!40000 ALTER TABLE `oc_t_item_stats` ENABLE KEYS */;
UNLOCK TABLES;
LOCK TABLES `oc_t_item_description` WRITE;
/*!40000 ALTER TABLE `oc_t_item_description` DISABLE KEYS */;
DELETE FROM `oc_t_item_description` WHERE fk_i_item_id >= 84143;
/*!40000 ALTER TABLE `oc_t_item_description` ENABLE KEYS */;
UNLOCK TABLES;
LOCK TABLES `oc_t_item` WRITE;
/*!40000 ALTER TABLE `oc_t_item` DISABLE KEYS */;
DELETE FROM `oc_t_item` WHERE pk_i_id >= 84143;
/*!40000 ALTER TABLE `oc_t_item` ENABLE KEYS */;
UNLOCK TABLES;
LOCK TABLES `oc_t_item_comment` WRITE;
/*!40000 ALTER TABLE `oc_t_item_comment` DISABLE KEYS */;
DELETE FROM `oc_t_item_comment`;
/*!40000 ALTER TABLE `oc_t_item_comment` ENABLE KEYS */;
UNLOCK TABLES;
LOCK TABLES `oc_t_user` WRITE;
/*!40000 ALTER TABLE `oc_t_user` DISABLE KEYS */;
DELETE FROM `oc_t_user`;
/*!40000 ALTER TABLE `oc_t_user` DISABLE KEYS */;
UNLOCK TABLES;


--
-- Dumping data for table `oc_t_user`
--

LOCK TABLES `oc_t_user` WRITE;
/*!40000 ALTER TABLE `oc_t_user` DISABLE KEYS */;
INSERT INTO `oc_t_user` VALUES (1,'2023-10-27 21:12:11','2023-10-27 21:12:11','Blake Sullivan','1','$2y$15$k/5QEKl24QCQjQYaJdjBO.KaGcVQ6lV24laSdHPgLEroaa3Ypn3Hy','HnqOjoBy','blake.sullivan@gmail.com','','','',1,1,NULL,NULL,NULL,NULL,'',NULL,'','',NULL,'',NULL,NULL,'',NULL,NULL,'',NULL,NULL,0,12,1,'2023-11-19 20:29:57','74.111.186.31',0,NULL,NULL);
/*!40000 ALTER TABLE `oc_t_user` ENABLE KEYS */;
UNLOCK TABLES;


-- Items
LOCK TABLES `oc_t_item` WRITE;
/*!40000 ALTER TABLE `oc_t_item` DISABLE KEYS */;
INSERT INTO `oc_t_item` VALUES
(84143,1,10,'2023-10-13 05:41:57','2023-10-13 05:41:57',NULL,27500000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'HFFjjEHx',0,0,0,'9999-12-31 23:59:59'),
(84144,1,10,'2023-11-14 17:18:36','2023-11-14 17:18:36',NULL,30000000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'FkB7WRYb',0,0,0,'9999-12-31 23:59:59'),
(84145,1,7,'2023-10-07 15:18:15',NULL,NULL,300000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'2tIsbJhW',0,0,0,'9999-12-31 23:59:59'),
(84146,1,7,'2023-11-09 08:38:12',NULL,NULL,250000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'9642UoV6',0,0,0,'9999-12-31 23:59:59'),
(84147,1,20,'2023-10-13 08:00:44',NULL,NULL,650000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'MC7BqqsV',0,1,0,'9999-12-31 23:59:59'),
(84148,1,18,'2023-11-04 04:48:41',NULL,NULL,80000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'5dlLurT4',0,1,0,'9999-12-31 23:59:59'),
(84149,1,18,'2023-10-22 16:41:44',NULL,NULL,120000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'W4WeNlfo',0,0,0,'9999-12-31 23:59:59'),
(84150,1,11,'2023-11-02 02:15:24',NULL,NULL,200000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'vgOmlnyQ',0,0,0,'9999-12-31 23:59:59'),
(84151,1,11,'2023-11-15 10:47:20',NULL,NULL,400000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'OvXiS9IY',0,0,0,'9999-12-31 23:59:59'),
(84152,1,10,'2023-10-17 11:34:43',NULL,NULL,5000000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'UcDSfl9a',0,1,0,'9999-12-31 23:59:59'),
(84153,1,3,'2023-11-07 05:52:37',NULL,NULL,1200000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'4JT22LcW',0,0,0,'9999-12-31 23:59:59'),
(84154,1,15,'2023-11-15 23:14:43',NULL,NULL,50000000,'USD','Blake Sullivan','blake.sullivan@gmail.com','','','74.111.186.31',0,1,1,0,'6zKpqIOk',0,1,0,'9999-12-31 23:59:59');
/*!40000 ALTER TABLE `oc_t_item` ENABLE KEYS */;
UNLOCK TABLES;


-- Dumping data for table `oc_t_item_location`
LOCK TABLES `oc_t_item_location` WRITE;
/*!40000 ALTER TABLE `oc_t_item_location` DISABLE KEYS */;
INSERT INTO `oc_t_item_location` VALUES
(84143,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8206379,'Pittsburgh',NULL,NULL,'',40.4406200000,-79.9958900000),
(84144,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8206379,'Pittsburgh',NULL,NULL,'',40.4406200000,-79.9958900000),
(84145,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8201734,'Monroeville',NULL,NULL,'',40.4211800000,-79.7881000000),
(84146,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8201734,'Monroeville',NULL,NULL,'',40.4211800000,-79.7881000000),
(84147,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8206379,'Pittsburgh',NULL,NULL,'',40.4406200000,-79.9958900000),
(84148,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8206379,'Pittsburgh',NULL,NULL,'',40.4406200000,-79.9958900000),
(84149,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8206379,'Pittsburgh',NULL,NULL,'',40.4406200000,-79.9958900000),
(84150,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8206379,'Pittsburgh',NULL,NULL,'',40.4406200000,-79.9958900000),
(84151,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8206379,'Pittsburgh',NULL,NULL,'',40.4406200000,-79.9958900000),
(84152,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8206379,'Pittsburgh',NULL,NULL,'',40.4406200000,-79.9958900000),
(84153,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8201734,'Monroeville',NULL,NULL,'',40.4211800000,-79.7881000000),
(84154,'US','United States','United States','',NULL,9254927,'Pennsylvania',NULL,8206379,'Pittsburgh',NULL,NULL,'',40.4406200000,-79.9958900000);
/*!40000 ALTER TABLE `oc_t_item_location` ENABLE KEYS */;
UNLOCK TABLES;


--
-- Dumping data for table `oc_t_item_resource`
--

LOCK TABLES `oc_t_item_resource` WRITE;
/*!40000 ALTER TABLE `oc_t_item_resource` DISABLE KEYS */;
-- DELETE FROM `oc_t_item_resource` WHERE pk_i_id IN (84143, 84144, 84145, 84146, 84147, 84148, 84149, 84150, 84151, 84152, 84153, 84154);
INSERT INTO `oc_t_item_resource` VALUES 
(84143,84143,'GOBTUniq','jpg','image/jpeg','oc-content/uploads/841/'),
(84144,84144,'CmMosAZu','jpg','image/jpeg','oc-content/uploads/841/'),
(84145,84145,'0CDGt7oz','jpg','image/jpeg','oc-content/uploads/841/'),
(84146,84146,'ocWoBHVt','jpg','image/jpeg','oc-content/uploads/841/'),
(84147,84147,'L4rvdckS','jpg','image/jpeg','oc-content/uploads/841/'),
(84148,84148,'3qIONABh','jpg','image/jpeg','oc-content/uploads/841/'),
(84149,84149,'YWIUjAtd','jpg','image/jpeg','oc-content/uploads/841/'),
(84150,84150,'apLyFw2n','jpg','image/jpeg','oc-content/uploads/841/'),
(84151,84151,'6iVB6WNW','jpg','image/jpeg','oc-content/uploads/841/'),
(84152,84152,'VkTAcMhb','jpg','image/jpeg','oc-content/uploads/841/'),
(84153,84153,'qzFNpL2p','jpg','image/jpeg','oc-content/uploads/841/'),
(84154,84154,'MvQy8UWD','jpg','image/jpeg','oc-content/uploads/841/');
/*!40000 ALTER TABLE `oc_t_item_resource` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `oc_t_item_description`
--
LOCK TABLES `oc_t_item_description` WRITE;
/*!40000 ALTER TABLE `oc_t_item_description` DISABLE KEYS */;
INSERT INTO `oc_t_item_description` VALUES 
(84143,'en_US','2019 Toyota 86 - Sporty Fun, Excellent Condition','Hit the road in style with this eye-catching 2019 Toyota 86. Known for its precise handling and brisk acceleration, this car has been a joy on both city streets and winding back roads. Kept in impeccable condition with just 15,000 miles on it, this vehicle has never seen a winter and has been garage-kept. Features include a responsive 6-speed manual transmission, rear-wheel drive, Bluetooth connectivity, and a premium sound system. Complete with a recent service, this car is ready to find its new owner who appreciates a blend of performance and reliability. Asking $27,500, serious inquiries only, please. Test drives available for interested buyers with proof of pre-approval or funds.'),
(84144,'en_US','Pristine 2021 Toyota 86 - Low Miles, Factory Warranty','Experience the thrill of driving with this pristine 2021 Toyota 86. With just 10,000 miles, this car still falls under the factory warranty, ensuring peace of mind along with excitement. Its 6-speed automatic transmission paired with a sport-tuned suspension provides a dynamic driving experience, while the clean, exterior and well-maintained interior speak volumes of its meticulous care. Features include Apple CarPlay, Android Auto, and a suite of safety features. Priced at $30,000, this beauty is looking for a driver who values style and substance. Available for viewing and test drives for serious buyers with financing pre-approved.'),
(84145,'en_US','Trail-Ready Hybrid Bike - 24-Speed, Lightweight Frame - $300','Get ready for adventure with this versatile hybrid bike, perfect for both city commutes and off-road trails. Featuring a lightweight aluminum frame, 24-speed gear system, and reliable hydraulic disc brakes for superior stopping power. The bike is in fantastic condition with very little wear on the tires and has been regularly serviced. I\'m including a set of safety lights and a lock with two keys for immediate use. Local pick-up available. Asking $300. Contact to schedule a test ride!'),
(84146,'en_US','Classic Road Bike - Lightweight Frame','Cruise the city in style with this classic road bike, featuring a stunning bright frame that stands out in any crowd. The bike is built with a lightweight and durable frame, offering a smooth and efficient ride. Equipped with a reliable gear set and drop handlebars for a true road cycling experience. It has been well-maintained and always stored indoors. Perfect for both the daily commuter or the weekend warrior. Asking $250. Contact me to come take a look and give it a test ride!'),
(84147,'en_US','Retro Scooter - City Cruiser, Great Mileage','Zip around town on this charming retro-style scooter, painted in a vibrant shade. It\'s a reliable and fun ride, perfect for quick errands or just enjoying a leisurely day out. Despite its classic look, it offers modern efficiency with great gas mileage and low maintenance costs. The scooter has a convenient storage basket, is easy to park, and a breeze to ride. It has been serviced regularly and kept in good condition. Available for $650. Helmet included.'),
(84148,'en_US','Modern Abstract Ceramic Vase - Unique Circular Design','For sale is a stunning modern ceramic vase with a unique circular design that creates a captivating visual interest. Its neutral color and matte finish make it a versatile piece for any decor style. This art piece stands out with its sculptural quality, perfect for those who appreciate modern art and clean lines. It\'s substantial in size and can serve as a standalone centerpiece or be paired with simple foliage for a natural aesthetic. Originally purchased from a boutique art gallery, it\'s now available for $80. Local pick up preferred.'),
(84149,'en_US','Stunning Glass Vase - Textured, Hand-Blown','Add a splash of color to your living space with this vibrant glass vase. This hand-blown piece features a unique bubbled texture that catches the light, creating a stunning visual effect. It\'s not just a vase, but a piece of art that stands out in any room. Ideal for holding fresh flowers or as a stand-alone decorative piece. Its round, bulbous body and narrow neck make it a striking addition to any tabletop or shelf.'),
(84150,'en_US','Space Gray iPhone 8 - 64GB, Unlocked, Like New','Up for sale is a Space Gray iPhone 8 with 64GB of storage. This phone is unlocked and ready to use with any carrier. It\'s in excellent condition, with no scratches or dents, as it\'s been in a case with a screen protector since day one. The battery life is still strong, and the phone operates smoothly without any issues. It comes with the original box and a charger. Great for someone looking for a reliable phone without the high price tag of newer models. Selling for $200. Local pickup or shipping available.'),
(84151,'en_US','iPhone X - 256GB, Space Gray, Excellent Condition','Selling an iPhone X in Space Gray with a generous 256GB of storage. This device has been meticulously maintained and comes with the latest iOS installed, ready for all your apps and media. The OLED screen delivers vibrant colors and true blacks for an immersive viewing experience. It features Face ID, dual rear cameras, and wireless charging capabilities. The phone has no scratches or dents and the battery health is at a strong performance level. Comes with the original box and charger. Selling for $400. Local pick up or can arrange for secure shipping.'),
(84152,'en_US','Classic 1995 Honda Civic, Well-Maintained','Here\'s a blast from the past with modern reliability – a classic 1995 Honda Civic. This car has been a treasured possession and is in remarkable condition for its age. It has covered a modest 120,000 miles and has been serviced regularly, with a detailed maintenance log available. The interior is clean, and the engine runs smoothly, reflecting the care given to it over the years. This Civic features manual transmission, offering the full control and driving experience enthusiasts seek. Asking price is $5,000.'),
(84153,'en_US','Premium Stainless Steel French Door Refrigerator - Spacious &amp; Modern','Upgrade your kitchen with this spacious and modern stainless steel French door refrigerator. Boasting a large capacity to accommodate all your storage needs, it features a convenient bottom freezer, adjustable shelves, and multiple storage bins. The sleek exterior with a water and ice dispenser adds to its sophisticated design, making it not only a practical appliance but also an elegant addition to any kitchen decor. With energy-efficient operation, this fridge is in excellent condition and ready to serve your household. Originally valued at over $2,000, it\'s now available for $1,200. Local delivery options available.'),
(84154,'en_US','Compact Canister Vacuum Cleaner - Lightweight and Powerful','Keep your home clean and dust-free with this gently used compact canister vacuum cleaner. Its lightweight design makes it easy to maneuver around furniture and tight spaces, while its powerful suction is perfect for all your floor types. The vacuum comes with a versatile hose and head that can tackle everything from hardwood floors to plush carpets. With its adjustable suction power, you can ensure delicate surfaces are treated gently. A great addition to any household looking for an efficient cleaning solution. Available for $50. Local pickup preferred, or delivery can be arranged for an additional fee.');
/*!40000 ALTER TABLE `oc_t_item_description` ENABLE KEYS */;
UNLOCK TABLES;


/*!40000 ALTER TABLE `oc_t_item_stats` DISABLE KEYS */;
INSERT INTO `oc_t_item_stats` VALUES 
(84143,0,0,0,0,0,0,0,'2023-11-19'),
(84144,0,0,0,0,0,0,0,'2023-11-19'),
(84145,0,0,0,0,0,0,0,'2023-11-19'),
(84146,0,0,0,0,0,0,0,'2023-11-19'),
(84147,0,0,0,0,0,0,0,'2023-11-19'),
(84148,0,0,0,0,0,0,0,'2023-11-19'),
(84149,0,0,0,0,0,0,0,'2023-11-19'),
(84150,0,0,0,0,0,0,0,'2023-11-19'),
(84151,0,0,0,0,0,0,0,'2023-11-19'),
(84152,0,0,0,0,0,0,0,'2023-11-19'),
(84153,0,0,0,0,0,0,0,'2023-11-19'),
(84154,0,0,0,0,0,0,0,'2023-11-19');
/*!40000 ALTER TABLE `oc_t_item_stats` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `oc_t_log`
--

LOCK TABLES `oc_t_log` WRITE;
/*!40000 ALTER TABLE `oc_t_log` DISABLE KEYS */;
INSERT INTO `oc_t_log` VALUES ('2023-11-02 02:54:53','item','add',84142,'Test post','74.111.186.31','user',1),
('2023-10-07 20:16:06','item','delete',84142,'Test post','74.111.186.31','user',1),
('2023-10-20 08:06:23','itemActions','deleteResourcesFromHD',84142,'84142','74.111.186.31','user',1),
('2023-11-10 07:18:05','item','delete resource',84142,'84142','74.111.186.31','user',1),
('2023-10-11 21:55:01','item','delete resource backtrace',84142,'#0 osc_deleteResource called@ [/usr/src/myapp/oc-includes/osclass/ItemActions.php:60] / #1 deleteResourcesFromHD called@ [/usr/src/myapp/oc-includes/osclass/model/Item.php:1054] / #2 deleteByPrimaryKey called@ [/usr/src/myapp/oc-includes/osclass/Item','74.111.186.31','user',1),
('2023-10-07 08:53:04','itemActions','deleteResourcesFromHD',84142,'84142,','74.111.186.31','user',1),
('2023-11-09 17:04:29','item','add',84143,'2019 Toyota 86 - Sleek Red, Sporty Fun, Excellent Condition','74.111.186.31','user',1),
('2023-10-21 16:52:40','item','add',84144,'Pristine 2021 Toyota 86 - Crisp White, Low Miles, Factory Warranty','74.111.186.31','user',1),
('2023-10-22 17:44:01','item','edit',84144,'Pristine 2021 Toyota 86 - Low Miles, Factory Warranty','74.111.186.31','user',1),
('2023-11-15 08:33:02','item','edit',84143,'2019 Toyota 86 - Sporty Fun, Excellent Condition','74.111.186.31','user',1),
('2023-10-13 05:41:57','item','add',84145,'Trail-Ready Hybrid Bike - 24-Speed, Lightweight Frame - $300','74.111.186.31','user',1),
('2023-11-14 17:18:36','item','add',84146,'Classic Road Bike - Lightweight Frame','74.111.186.31','user',1),
('2023-10-07 15:18:15','item','add',84147,'Retro Scooter - City Cruiser, Great Mileage','74.111.186.31','user',1),
('2023-11-09 08:38:12','item','add',84148,'Modern Abstract Ceramic Vase - Unique Circular Design','74.111.186.31','user',1),
('2023-10-13 08:00:44','item','add',84149,'Stunning Glass Vase - Textured, Hand-Blown','74.111.186.31','user',1),
('2023-11-04 04:48:41','item','add',84150,'Space Gray iPhone 8 - 64GB, Unlocked, Like New','74.111.186.31','user',1),
('2023-10-22 16:41:44','item','add',84151,'iPhone X - 256GB, Space Gray, Excellent Condition','74.111.186.31','user',1),
('2023-11-02 02:15:24','item','edit',84144,'Pristine 2021 Toyota 86 - Low Miles, Factory Warranty','74.111.186.31','user',1),
('2023-11-15 10:47:20','item','edit',84143,'2019 Toyota 86 - Sporty Fun, Excellent Condition','74.111.186.31','user',1),
('2023-10-17 11:34:43','item','add',84152,'Classic 1995 Honda Civic, Well-Maintained','74.111.186.31','user',1),
('2023-11-07 05:52:37','item','add',84153,'Premium Stainless Steel French Door Refrigerator - Spacious &amp; Modern','74.111.186.31','user',1),
('2023-11-15 23:14:43','item','add',84154,'Compact Canister Vacuum Cleaner - Lightweight and Powerful','74.111.186.31','user',1);
/*!40000 ALTER TABLE `oc_t_log` ENABLE KEYS */;
UNLOCK TABLES;
